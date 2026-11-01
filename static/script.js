document.addEventListener('DOMContentLoaded', () => {
    const input = document.getElementById('text-input');
    // Analysis History elements
    const historyList = document.getElementById('history-list');
    const historyLoading = document.getElementById('history-loading');
    const historyEmpty = document.getElementById('history-empty');
    const historyError = document.getElementById('history-error');
    const historyErrorText = document.getElementById('history-error-text');
    const refreshHistoryBtn = document.getElementById('refresh-history-btn');
    const charCount = document.getElementById('char-count');
    const analyzeBtn = document.getElementById('analyze-btn');
    const btnText = document.querySelector('.btn-text');
    const btnIcon = document.querySelector('.btn-icon');
    const spinner = document.getElementById('loading-spinner');
    const resultContainer = document.getElementById('result-container');
    const resultBox = document.getElementById('result-box');
    const resultEmoji = document.getElementById('result-emoji');
    const resultText = document.getElementById('result-text');
    const errorContainer = document.getElementById('error-container');
    const errorText = document.getElementById('error-text');
    const dynamicGraphContainer = document.getElementById('dynamic-graph-container');
    const explanationContainer = document.getElementById('explanation-container');
    const explanationText = document.getElementById('explanation-text');
    const tokenContainer = document.getElementById('token-container');
    const tokenList = document.getElementById('token-list');
    let confidenceChart = null;


    // Batch elements
    const batchInput = document.getElementById('batch-input');
    const batchCount = document.getElementById('batch-count');
    const batchAnalyzeBtn = document.getElementById('batch-analyze-btn');
    const batchLoadingSpinner = document.getElementById('batch-loading-spinner');

    const batchResultContainer = document.getElementById('batch-result-container');
    const batchTotal = document.getElementById('batch-total');
    const batchResults = document.getElementById('batch-results');
    // Advanced batch analytics elements
    const batchAverageConfidence = document.getElementById('batch-average-confidence');

    const mostPositiveText = document.getElementById('most-positive-text');

    const mostPositiveConfidence = document.getElementById('most-positive-confidence');

    const mostNegativeText = document.getElementById('most-negative-text');

    const mostNegativeConfidence = document.getElementById('most-negative-confidence');

    const batchErrorContainer = document.getElementById('batch-error-container');
    const batchErrorText = document.getElementById('batch-error-text');

    let batchSentimentChart = null;
    const exportCsvBtn = document.getElementById('export-csv-btn');

    let latestBatchResults = [];

    batchInput.addEventListener('input', () => {

        const tweets = batchInput.value
            .split('\n')
            .map(tweet => tweet.trim())
            .filter(tweet => tweet.length > 0);

        batchCount.textContent =
            `${tweets.length} tweet${tweets.length === 1 ? '' : 's'}`;

    });

    batchAnalyzeBtn.addEventListener('click', async () => {

        const texts = batchInput.value
            .split('\n')
            .map(tweet => tweet.trim())
            .filter(tweet => tweet.length > 0);

        batchErrorContainer.classList.add('hidden');
        batchResultContainer.classList.add('hidden');

        if (texts.length === 0) {
            showBatchError("Please enter at least one tweet.");
            return;
        }

        if (texts.length > 50) {
            showBatchError("Maximum 50 tweets can be analyzed at once.");
            return;
        }

        batchAnalyzeBtn.disabled = true;

        batchAnalyzeBtn.querySelector('.btn-text').textContent =
            "Analyzing...";

        batchLoadingSpinner.classList.remove('hidden');

        try {

            const response = await fetch('/api/batch-predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    texts: texts
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(
                    data.error || "Batch analysis failed."
                );
            }

            displayBatchResults(data);

        } catch (error) {

            console.error("Batch analysis error:", error);

            showBatchError(
                error.message || "An error occurred while analyzing the tweets."
            );

        } finally {

            batchAnalyzeBtn.disabled = false;

            batchAnalyzeBtn.querySelector('.btn-text').textContent =
                "Analyze Batch";

            batchLoadingSpinner.classList.add('hidden');
        }

    });
    function displayBatchResults(data) {

        latestBatchResults = data.results;

        batchResultContainer.classList.remove('hidden');

        batchTotal.textContent = data.total;

        // Display average confidence
        batchAverageConfidence.textContent =
            `${data.average_confidence.toFixed(2)}%`;

        // Display strongest positive tweet
        const positive = data.strongest_examples.most_positive;

        if (positive) {
            mostPositiveText.textContent = positive.text;

            mostPositiveConfidence.textContent =
                `Positive confidence: ${(positive.probabilities.Positive * 100).toFixed(2)}%`;
        }
        else {
            mostPositiveText.textContent = "No positive tweets found.";
            mostPositiveConfidence.textContent = "";
        }

        // Display strongest negative tweet
        const negative = data.strongest_examples.most_negative;

        if (negative) {
            mostNegativeText.textContent = negative.text;

            mostNegativeConfidence.textContent =
                `Negative confidence: ${(negative.probabilities.Negative * 100).toFixed(2)}%`;
        }
        else {
            mostNegativeText.textContent = "No negative tweets found    .";
            mostNegativeConfidence.textContent = "";
        }

        displayBatchChart(data.sentiment_counts);

        batchResults.innerHTML = '';

        data.results.forEach((result, index) => {

            const resultElement = document.createElement('div');

            resultElement.classList.add('batch-result-item');

            resultElement.innerHTML = `
                <div class="batch-result-number">
                    ${index + 1}
                </div>

                <div class="batch-result-content">

                    <p class="batch-result-text">
                        ${escapeHtml(result.text)}
                    </p>

                    <span class="batch-result-sentiment ${result.sentiment.toLowerCase()}">
                        ${result.sentiment}
                    </span>

                </div>
            `;

            batchResults.appendChild(resultElement);
        });
    }
    function displayBatchChart(sentimentCounts) {

        const ctx = document.getElementById(
            'batch-sentiment-chart'
        ).getContext('2d');

        if (batchSentimentChart) {
            batchSentimentChart.destroy();
        }

        batchSentimentChart = new Chart(ctx, {

            type: 'doughnut',

            data: {
                labels: [
                    'Negative',
                    'Neutral',
                    'Positive'
                ],

                datasets: [{
                    data: [
                        sentimentCounts.Negative,
                        sentimentCounts.Neutral,
                        sentimentCounts.Positive
                    ],

                    borderWidth: 0
                }]
            },

            options: {

                responsive: true,

                maintainAspectRatio: false,

                plugins: {

                    legend: {
                        position: 'bottom'
                    },

                    tooltip: {
                        callbacks: {
                            label: function (context) {

                                const value = context.raw;

                                const total =
                                    sentimentCounts.Negative +
                                    sentimentCounts.Neutral +
                                    sentimentCounts.Positive;

                                const percentage =
                                    ((value / total) * 100).toFixed(1);

                                return `${context.label}: ${value} (${percentage}%)`;
                            }
                        }
                    }
                }
            }

        });
    }
    function showBatchError(message) {

        batchErrorText.textContent = message;

        batchErrorContainer.classList.remove('hidden');
    }


    // Export batch results as CSV
    exportCsvBtn.addEventListener('click', exportBatchCSV);

    function exportBatchCSV() {

        if (latestBatchResults.length === 0) {
            alert("No batch results available to export.");
            return;
        }

        const headers = [
            "Tweet",
            "Predicted Sentiment",
            "Negative Probability",
            "Neutral Probability",
            "Positive Probability",
            "Prediction Confidence"
        ];

        const rows = latestBatchResults.map(result => {

            const probabilities = result.probabilities;

            const confidence = Math.max(
                probabilities.Negative,
                probabilities.Neutral,
                probabilities.Positive
            );

            return [
                result.text,
                result.sentiment,
                probabilities.Negative,
                probabilities.Neutral,
                probabilities.Positive,
                confidence
            ];
        });

        // Escape CSV values safely
        const escapeCsv = value => {
            let text = String(value ?? "");

            // Prevent spreadsheet formula execution from untrusted text
            if (/^[\s]*[=+\-@]/.test(text)) {
                text = "'" + text;
            }

            return `"${text.replace(/"/g, '""')}"`;
        };

        const csvContent = [
            headers.map(escapeCsv).join(","),
            ...rows.map(row => row.map(escapeCsv).join(","))
        ].join("\r\n");

        const blob = new Blob(
            ["\uFEFF" + csvContent],
            { type: "text/csv;charset=utf-8;" }
        );

        const url = URL.createObjectURL(blob);

        const link = document.createElement("a");

        link.href = url;
        link.download = "sentiment_analysis_results.csv";

        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

        URL.revokeObjectURL(url);
    }


    function escapeHtml(text) {

        const div = document.createElement('div');

        div.textContent = text;

        return div.innerHTML;
    }

    // Handle Character Count
    input.addEventListener('input', () => {
        const length = input.value.length;
        charCount.textContent = length;

        if (length > 280) {
            charCount.style.color = '#ef4444';
        } else {
            charCount.style.color = '#94a3b8';
        }
    });

    analyzeBtn.addEventListener('click', async () => {
        const text = input.value.trim();

        if (!text) {
            showError("Please enter some text to analyze.");
            return;
        }

        // Reset state
        hideError();
        resultContainer.classList.add('hidden');
        resultContainer.classList.remove('active');
        setLoading(true);

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Something went wrong during analysis.');
            }

            showResult(
                data.sentiment,
                data.probabilities,
                data.explanation,
                data.token_importance
            );
        } catch (err) {
            showError(err.message);
        } finally {
            setLoading(false);
        }
    });

    function setLoading(isLoading) {
        if (isLoading) {
            analyzeBtn.disabled = true;
            btnText.textContent = "Analyzing...";
            btnIcon.classList.add('hidden');
            spinner.classList.remove('hidden');
        } else {
            analyzeBtn.disabled = false;
            btnText.textContent = "Analyze Now";
            btnIcon.classList.remove('hidden');
            spinner.classList.add('hidden');
        }
    }

    function showResult(sentiment, probabilities, explanation, tokenImportance) {
        resultContainer.classList.remove('hidden');
        resultContainer.classList.add('active');

        // Remove existing classes
        resultBox.classList.remove('positive', 'negative', 'neutral');

        resultText.textContent = sentiment;

        if (sentiment === 'Positive') {
            resultBox.classList.add('positive');
            resultEmoji.textContent = '😃';
        } else if (sentiment === 'Negative') {
            resultBox.classList.add('negative');
            resultEmoji.textContent = '😠';
        } else if (sentiment === 'Neutral') {
            resultBox.classList.add('neutral');
            resultEmoji.textContent = '😐';
        } else {
            resultEmoji.textContent = '🤔';
        }

        // Update Chart
        if (probabilities) {
            renderChart(probabilities);
            dynamicGraphContainer.classList.remove('hidden');
        } else {
            dynamicGraphContainer.classList.add('hidden');
        }
        // Update explanation
        if (explanation) {
            explanationText.textContent = explanation;
            explanationContainer.classList.remove('hidden');
        } else {
            explanationContainer.classList.add('hidden');
        }
        // Display important tokens
        if (tokenImportance && tokenImportance.length > 0) {
            renderTokenImportance(tokenImportance);
            tokenContainer.classList.remove('hidden');
        } else {
            tokenContainer.classList.add('hidden');
        }

        // Trigger a slight animation reflow
        resultContainer.style.animation = 'none';
        resultContainer.offsetHeight; /* trigger reflow */
        resultContainer.style.animation = null;
    }

    function renderChart(probs) {
        const ctx = document.getElementById('confidence-chart').getContext('2d');

        const labels = ['Negative', 'Neutral', 'Positive'];
        const dataValues = [
            (probs.Negative * 100).toFixed(1),
            (probs.Neutral * 100).toFixed(1),
            (probs.Positive * 100).toFixed(1)
        ];

        const colors = ['#ef4444', '#f59e0b', '#10b981'];

        if (confidenceChart) {
            confidenceChart.destroy();
        }

        confidenceChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    data: dataValues,
                    backgroundColor: colors,
                    borderRadius: 8,
                    barThickness: 24,
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (context) => ` ${context.raw}%`
                        }
                    }
                },
                scales: {
                    x: {
                        display: false,
                        max: 100,
                        beginAtZero: true
                    },
                    y: {
                        grid: { display: false },
                        ticks: {
                            color: '#94a3b8',
                            font: {
                                family: 'Inter',
                                size: 14,
                                weight: '600'
                            }
                        }
                    }
                },
                animation: {
                    duration: 1000,
                    easing: 'easeOutQuart'
                }
            }
        });
    }
    function renderTokenImportance(tokens) {
        tokenList.innerHTML = '';

        // Display only the 10 most influential tokens
        tokens.slice(0, 10).forEach(item => {
            const token = item.token
                .replace('Ġ', '')
                .trim();

            if (!token) return;

            const importance = item.importance;

            const tokenElement = document.createElement('span');

            tokenElement.classList.add('importance-token');

            if (importance >= 0) {
                tokenElement.classList.add('positive-token');
            } else {
                tokenElement.classList.add('negative-token');
            }

            tokenElement.textContent = token;

            tokenElement.title =
                `Importance: ${importance.toFixed(4)}`;

            tokenList.appendChild(tokenElement);
        });
    }

    function showError(message) {
        errorText.textContent = message;
        errorContainer.classList.remove('hidden');
    }

    function hideError() {
        errorContainer.classList.add('hidden');
    }


    // =====================================
    // ANALYSIS HISTORY
    // =====================================

    async function loadHistory() {

        historyLoading.classList.remove('hidden');
        historyEmpty.classList.add('hidden');
        historyError.classList.add('hidden');
        historyList.innerHTML = '';

        try {

            const response = await fetch('/api/history');
            const data = await response.json();

            if (!response.ok) {
                throw new Error(
                    data.error || 'Unable to load analysis history.'
                );
            }

            historyLoading.classList.add('hidden');

            if (!data.history || data.history.length === 0) {
                historyEmpty.classList.remove('hidden');
                return;
            }

            displayHistory(data.history);

        } catch (error) {

            console.error('History loading error:', error);

            historyLoading.classList.add('hidden');
            historyErrorText.textContent = error.message;
            historyError.classList.remove('hidden');

        }
    }


    function displayHistory(records) {

        historyList.innerHTML = '';

        records.forEach(record => {

            const item = document.createElement('div');
            item.classList.add('history-item');

            const sentiment = record.sentiment.toLowerCase();

            const probabilities = record.probabilities;

            const date = new Date(record.created_at);

            const formattedDate = date.toLocaleString();

            item.innerHTML = `
                <div class="history-item-main">

                    <div class="history-item-content">

                        <p class="history-item-text">
                            ${escapeHtml(record.text)}
                        </p>

                        <div class="history-item-meta">

                            <span class="history-sentiment ${sentiment}">
                                ${escapeHtml(record.sentiment)}
                            </span>

                            <span class="history-date">
                                ${escapeHtml(formattedDate)}
                            </span>

                        </div>

                        <div class="history-probabilities">

                            <span>
                                Negative:
                                ${(probabilities.Negative * 100).toFixed(1)}%
                            </span>

                            <span>
                                Neutral:
                                ${(probabilities.Neutral * 100).toFixed(1)}%
                            </span>

                            <span>
                                Positive:
                                ${(probabilities.Positive * 100).toFixed(1)}%
                            </span>

                        </div>

                    </div>

                    <button
                        class="delete-history-btn"
                        data-id="${Number(record.id)}"
                        title="Delete this record">

                        <i data-lucide="trash-2"></i>

                    </button>

                </div>
            `;

            historyList.appendChild(item);

        });

        // Initialize icons for dynamically added elements
        lucide.createIcons();
    }


    async function deleteHistory(id) {

        const confirmed = confirm(
            'Are you sure you want to delete this analysis record?'
        );

        if (!confirmed) return;

        try {

            const response = await fetch(`/api/history/${id}`, {
                method: 'DELETE'
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(
                    data.error || 'Unable to delete history record.'
                );
            }

            loadHistory();

        } catch (error) {

            console.error('History deletion error:', error);

            historyErrorText.textContent = error.message;
            historyError.classList.remove('hidden');

        }
    }


    // Handle delete button clicks
    historyList.addEventListener('click', event => {

        const deleteButton = event.target.closest(
            '.delete-history-btn'
        );

        if (!deleteButton) return;

        const id = Number(deleteButton.dataset.id);

        if (Number.isInteger(id) && id > 0) {
            deleteHistory(id);
        }

    });


    // Refresh history button
    refreshHistoryBtn.addEventListener('click', loadHistory);


    // Load history when the page opens
    loadHistory();


});

