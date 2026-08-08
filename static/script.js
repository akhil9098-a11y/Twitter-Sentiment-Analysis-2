document.addEventListener('DOMContentLoaded', () => {
    const input = document.getElementById('text-input');
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
    let confidenceChart = null;

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

            showResult(data.sentiment, data.probabilities);
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

    function showResult(sentiment, probabilities) {
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

    function showError(message) {
        errorText.textContent = message;
        errorContainer.classList.remove('hidden');
    }
    
    function hideError() {
        errorContainer.classList.add('hidden');
    }
});
