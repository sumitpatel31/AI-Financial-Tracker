/**
 * Enhanced Chart.js Configurations for AI-Powered Financial Manager
 */

// Global Chart.js Defaults for Dark Theme
Chart.defaults.color = 'rgba(255, 255, 255, 0.9)';
Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.15)';
Chart.defaults.backgroundColor = 'rgba(255, 255, 255, 0.05)';
Chart.defaults.font.family = 'Inter, system-ui, sans-serif';
Chart.defaults.font.weight = '500';
Chart.defaults.font.size = 12;

// Professional Color Palettes
const ChartColors = {
    primary: [
        '#00d4ff', '#1dd1ff', '#3acedf', '#57ccbf',
        '#74c69d', '#90e0ef', '#a8dadc', '#caf0f8'
    ],
    success: [
        '#00ff88', '#20ff99', '#40ffaa', '#60ffbb',
        '#80ffcc', '#a0ffdd', '#c0ffee', '#e0ffff'
    ],
    danger: [
        '#ff3366', '#ff4d77', '#ff6688', '#ff8099',
        '#ff99aa', '#ffb3bb', '#ffcccc', '#ffe6dd'
    ],
    warning: [
        '#ffaa00', '#ffb41a', '#ffbe33', '#ffc84d',
        '#ffd166', '#ffdb80', '#ffe599', '#ffefb3'
    ],
    info: [
        '#8c7ae6', '#9688ea', '#a096ee', '#aaa4f2',
        '#b4b2f6', '#bec0fa', '#c8cefe', '#d2dcff'
    ],
    accent: [
        '#ff6b6b', '#ff7979', '#ff8787', '#ff9595',
        '#ffa3a3', '#ffb1b1', '#ffbfbf', '#ffcdcd'
    ],
    gradients: [
        'linear-gradient(135deg, #00d4ff 0%, #8c7ae6 100%)',
        'linear-gradient(135deg, #ff6b6b 0%, #00ff88 100%)',
        'linear-gradient(135deg, #ffaa00 0%, #ff3366 100%)',
        'linear-gradient(135deg, #8c7ae6 0%, #00d4ff 100%)',
        'linear-gradient(135deg, #00ff88 0%, #ffaa00 100%)',
        'linear-gradient(135deg, #ff3366 0%, #8c7ae6 100%)'
    ],
    glassmorphism: [
        'rgba(0, 212, 255, 0.3)',
        'rgba(255, 107, 107, 0.3)',
        'rgba(0, 255, 136, 0.3)',
        'rgba(255, 170, 0, 0.3)',
        'rgba(140, 122, 230, 0.3)',
        'rgba(255, 51, 102, 0.3)'
    ]
};

// Chart Animation Configurations
const ChartAnimations = {
    smooth: {
        duration: 1500,
        easing: 'easeInOutQuart'
    },
    bounce: {
        duration: 2000,
        easing: 'easeOutBounce'
    },
    elastic: {
        duration: 2500,
        easing: 'easeOutElastic'
    }
};

// Common Chart Options
const CommonChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
        intersect: false,
        mode: 'index'
    },
    plugins: {
        legend: {
            position: 'top',
            labels: {
                usePointStyle: true,
                padding: 20,
                font: {
                    size: 11,
                    weight: '600'
                },
                color: 'rgba(255, 255, 255, 0.9)'
            }
        },
        tooltip: {
            backgroundColor: 'rgba(26, 26, 26, 0.95)',
            titleColor: '#00d4ff',
            bodyColor: 'rgba(255, 255, 255, 0.9)',
            borderColor: 'rgba(0, 212, 255, 0.3)',
            borderWidth: 1,
            cornerRadius: 12,
            padding: 16,
            titleFont: {
                size: 14,
                weight: '600'
            },
            bodyFont: {
                size: 13
            },
            displayColors: true,
            usePointStyle: true
        }
    },
    scales: {
        x: {
            grid: {
                color: 'rgba(255, 255, 255, 0.1)',
                lineWidth: 1
            },
            ticks: {
                color: 'rgba(255, 255, 255, 0.8)',
                font: {
                    size: 11
                }
            }
        },
        y: {
            grid: {
                color: 'rgba(255, 255, 255, 0.1)',
                lineWidth: 1
            },
            ticks: {
                color: 'rgba(255, 255, 255, 0.8)',
                font: {
                    size: 11
                },
                callback: function(value) {
                    return '₹' + value.toLocaleString();
                }
            }
        }
    }
};

/**
 * Enhanced Line Chart Creator
 */
function createAdvancedLineChart(ctx, data, options = {}) {
    const defaultOptions = {
        ...CommonChartOptions,
        elements: {
            line: {
                tension: 0.4,
                borderWidth: 3,
                borderCapStyle: 'round',
                borderJoinStyle: 'round'
            },
            point: {
                radius: 6,
                hoverRadius: 8,
                borderWidth: 2,
                hoverBorderWidth: 3
            }
        },
        plugins: {
            ...CommonChartOptions.plugins,
            filler: {
                propagate: false
            }
        },
        animation: ChartAnimations.smooth
    };

    return new Chart(ctx, {
        type: 'line',
        data: data,
        options: { ...defaultOptions, ...options }
    });
}

/**
 * Enhanced Doughnut Chart Creator
 */
function createAdvancedDoughnutChart(ctx, data, options = {}) {
    const defaultOptions = {
        ...CommonChartOptions,
        cutout: '70%',
        plugins: {
            ...CommonChartOptions.plugins,
            legend: {
                position: 'bottom',
                labels: {
                    usePointStyle: true,
                    padding: 20,
                    font: {
                        size: 11,
                        weight: '500'
                    }
                }
            }
        },
        elements: {
            arc: {
                borderWidth: 3,
                borderColor: 'rgba(26, 26, 26, 0.8)',
                hoverBorderWidth: 4
            }
        },
        animation: {
            ...ChartAnimations.bounce,
            animateRotate: true,
            animateScale: true
        }
    };

    return new Chart(ctx, {
        type: 'doughnut',
        data: data,
        options: { ...defaultOptions, ...options }
    });
}

/**
 * Enhanced Bar Chart Creator
 */
function createAdvancedBarChart(ctx, data, options = {}) {
    const defaultOptions = {
        ...CommonChartOptions,
        elements: {
            bar: {
                borderRadius: 8,
                borderSkipped: false,
                borderWidth: 2,
                borderColor: 'transparent'
            }
        },
        plugins: {
            ...CommonChartOptions.plugins,
            legend: {
                display: false
            }
        },
        animation: {
            ...ChartAnimations.elastic,
            delay: (context) => {
                return context.dataIndex * 100;
            }
        }
    };

    return new Chart(ctx, {
        type: 'bar',
        data: data,
        options: { ...defaultOptions, ...options }
    });
}

/**
 * Advanced Radar Chart Creator
 */
function createAdvancedRadarChart(ctx, data, options = {}) {
    const defaultOptions = {
        responsive: true,
        maintainAspectRatio: false,
        elements: {
            line: {
                borderWidth: 3,
                tension: 0.1
            },
            point: {
                borderWidth: 2,
                radius: 6,
                hoverRadius: 8
            }
        },
        plugins: {
            legend: {
                position: 'top',
                labels: {
                    usePointStyle: true,
                    padding: 20,
                    color: 'rgba(255, 255, 255, 0.9)'
                }
            },
            tooltip: CommonChartOptions.plugins.tooltip
        },
        scales: {
            r: {
                angleLines: {
                    color: 'rgba(255, 255, 255, 0.2)'
                },
                grid: {
                    color: 'rgba(255, 255, 255, 0.2)'
                },
                pointLabels: {
                    color: 'rgba(255, 255, 255, 0.8)',
                    font: {
                        size: 11
                    }
                },
                ticks: {
                    color: 'rgba(255, 255, 255, 0.6)',
                    backdropColor: 'transparent'
                }
            }
        },
        animation: ChartAnimations.smooth
    };

    return new Chart(ctx, {
        type: 'radar',
        data: data,
        options: { ...defaultOptions, ...options }
    });
}

/**
 * Financial Health Score Gauge Chart
 */
function createHealthScoreChart(ctx, score, maxScore = 100) {
    const percentage = (score / maxScore) * 100;
    
    // Determine color based on score
    let color = ChartColors.danger[0];
    if (percentage >= 80) color = ChartColors.success[0];
    else if (percentage >= 60) color = ChartColors.warning[0];
    else if (percentage >= 40) color = ChartColors.info[0];

    const data = {
        datasets: [{
            data: [percentage, 100 - percentage],
            backgroundColor: [color, 'rgba(255, 255, 255, 0.1)'],
            borderWidth: 0,
            cutout: '80%'
        }]
    };

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: false
            },
            tooltip: {
                enabled: false
            }
        },
        animation: {
            animateRotate: true,
            duration: 2000,
            easing: 'easeOutQuart'
        }
    };

    return new Chart(ctx, {
        type: 'doughnut',
        data: data,
        options: options
    });
}

/**
 * Trend Analysis Chart with Multiple Datasets
 */
function createTrendAnalysisChart(ctx, trendData, period = 6) {
    const displayData = trendData.slice(-period);
    
    const data = {
        labels: displayData.map(d => d.short_month),
        datasets: [{
            label: 'Income',
            data: displayData.map(d => d.income),
            borderColor: ChartColors.success[0],
            backgroundColor: ChartColors.success[0] + '20',
            tension: 0.4,
            fill: true,
            pointBackgroundColor: ChartColors.success[0],
            pointBorderColor: '#fff',
            pointBorderWidth: 2,
            pointRadius: 6,
            pointHoverRadius: 8
        }, {
            label: 'Expenses',
            data: displayData.map(d => d.expenses),
            borderColor: ChartColors.danger[0],
            backgroundColor: ChartColors.danger[0] + '20',
            tension: 0.4,
            fill: true,
            pointBackgroundColor: ChartColors.danger[0],
            pointBorderColor: '#fff',
            pointBorderWidth: 2,
            pointRadius: 6,
            pointHoverRadius: 8
        }, {
            label: 'Savings',
            data: displayData.map(d => d.savings),
            borderColor: ChartColors.primary[0],
            backgroundColor: ChartColors.primary[0] + '20',
            tension: 0.4,
            fill: true,
            pointBackgroundColor: ChartColors.primary[0],
            pointBorderColor: '#fff',
            pointBorderWidth: 2,
            pointRadius: 6,
            pointHoverRadius: 8
        }]
    };

    return createAdvancedLineChart(ctx, data, {
        plugins: {
            ...CommonChartOptions.plugins,
            title: {
                display: true,
                text: `Financial Trend (${period} Months)`,
                color: 'rgba(255, 255, 255, 0.9)',
                font: {
                    size: 16,
                    weight: '600'
                }
            }
        }
    });
}

/**
 * Category Spending Breakdown
 */
function createCategoryBreakdownChart(ctx, categoryData) {
    const data = {
        labels: Object.keys(categoryData),
        datasets: [{
            data: Object.values(categoryData),
            backgroundColor: ChartColors.primary.concat(
                ChartColors.success,
                ChartColors.warning,
                ChartColors.info
            ),
            borderWidth: 3,
            borderColor: 'rgba(26, 26, 26, 0.8)',
            hoverBorderWidth: 4,
            hoverBorderColor: '#fff'
        }]
    };

    return createAdvancedDoughnutChart(ctx, data, {
        plugins: {
            ...CommonChartOptions.plugins,
            legend: {
                position: 'bottom',
                labels: {
                    padding: 15,
                    usePointStyle: true,
                    font: {
                        size: 10
                    }
                }
            }
        }
    });
}

/**
 * Monthly Comparison Chart
 */
function createMonthlyComparisonChart(ctx, comparisonData) {
    const data = {
        labels: comparisonData.map(d => d.month),
        datasets: [{
            label: 'This Year',
            data: comparisonData.map(d => d.thisYear),
            backgroundColor: ChartColors.primary[0],
            borderColor: ChartColors.primary[0],
            borderWidth: 2,
            borderRadius: 8,
            borderSkipped: false
        }, {
            label: 'Last Year',
            data: comparisonData.map(d => d.lastYear),
            backgroundColor: ChartColors.info[0],
            borderColor: ChartColors.info[0],
            borderWidth: 2,
            borderRadius: 8,
            borderSkipped: false
        }]
    };

    return createAdvancedBarChart(ctx, data, {
        plugins: {
            ...CommonChartOptions.plugins,
            legend: {
                display: true,
                position: 'top'
            }
        },
        scales: {
            ...CommonChartOptions.scales,
            x: {
                ...CommonChartOptions.scales.x,
                stacked: false
            },
            y: {
                ...CommonChartOptions.scales.y,
                stacked: false
            }
        }
    });
}

/**
 * Expense Distribution by Day of Week
 */
function createExpenseDistributionChart(ctx, distributionData) {
    const data = {
        labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        datasets: [{
            label: 'Average Spending',
            data: distributionData,
            backgroundColor: ChartColors.primary.slice(0, 7),
            borderWidth: 2,
            borderColor: 'rgba(255, 255, 255, 0.2)',
            borderRadius: 8,
            borderSkipped: false
        }]
    };

    return createAdvancedBarChart(ctx, data, {
        indexAxis: 'y',
        plugins: {
            ...CommonChartOptions.plugins,
            legend: {
                display: false
            }
        }
    });
}

/**
 * Savings Goal Progress Chart
 */
function createSavingsProgressChart(ctx, current, target) {
    const percentage = Math.min((current / target) * 100, 100);
    
    const data = {
        datasets: [{
            data: [percentage, 100 - percentage],
            backgroundColor: [
                percentage >= 100 ? ChartColors.success[0] : ChartColors.warning[0],
                'rgba(255, 255, 255, 0.1)'
            ],
            borderWidth: 0,
            cutout: '75%'
        }]
    };

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: false
            },
            tooltip: {
                callbacks: {
                    label: function(context) {
                        return `Progress: ${percentage.toFixed(1)}%`;
                    }
                }
            }
        },
        animation: {
            animateRotate: true,
            duration: 2000,
            easing: 'easeOutQuart'
        }
    };

    return new Chart(ctx, {
        type: 'doughnut',
        data: data,
        options: options
    });
}

/**
 * Utility Functions
 */
const ChartUtils = {
    // Format currency for tooltips
    formatCurrency: (value) => {
        return new Intl.NumberFormat('en-IN', {
            style: 'currency',
            currency: 'INR',
            minimumFractionDigits: 0,
            maximumFractionDigits: 0
        }).format(value);
    },

    // Format percentage
    formatPercentage: (value) => {
        return value.toFixed(1) + '%';
    },

    // Generate gradient for canvas
    createGradient: (ctx, color1, color2, vertical = false) => {
        const gradient = vertical 
            ? ctx.createLinearGradient(0, 0, 0, ctx.canvas.height)
            : ctx.createLinearGradient(0, 0, ctx.canvas.width, 0);
        
        gradient.addColorStop(0, color1);
        gradient.addColorStop(1, color2);
        return gradient;
    },

    // Add loading state
    showLoading: (element) => {
        element.classList.add('loading');
        element.innerHTML = '<div class="text-center py-5"><div class="spinner-border text-primary" role="status"></div></div>';
    },

    // Remove loading state
    hideLoading: (element) => {
        element.classList.remove('loading');
    },

    // Animate chart appearance
    animateChart: (chart, delay = 0) => {
        setTimeout(() => {
            chart.update('active');
        }, delay);
    },

    // Destroy chart safely
    destroyChart: (chart) => {
        if (chart && typeof chart.destroy === 'function') {
            chart.destroy();
        }
    }
};

// Global chart instances storage
const ChartInstances = new Map();

/**
 * Chart Manager for handling multiple charts
 */
const ChartManager = {
    register: (id, chart) => {
        if (ChartInstances.has(id)) {
            ChartUtils.destroyChart(ChartInstances.get(id));
        }
        ChartInstances.set(id, chart);
    },

    get: (id) => {
        return ChartInstances.get(id);
    },

    destroy: (id) => {
        const chart = ChartInstances.get(id);
        if (chart) {
            ChartUtils.destroyChart(chart);
            ChartInstances.delete(id);
        }
    },

    destroyAll: () => {
        ChartInstances.forEach((chart, id) => {
            ChartUtils.destroyChart(chart);
        });
        ChartInstances.clear();
    },

    update: (id, newData) => {
        const chart = ChartInstances.get(id);
        if (chart) {
            chart.data = newData;
            chart.update('active');
        }
    }
};

// Export for global use
window.ChartColors = ChartColors;
window.ChartAnimations = ChartAnimations;
window.CommonChartOptions = CommonChartOptions;
window.ChartUtils = ChartUtils;
window.ChartManager = ChartManager;
window.createAdvancedLineChart = createAdvancedLineChart;
window.createAdvancedDoughnutChart = createAdvancedDoughnutChart;
window.createAdvancedBarChart = createAdvancedBarChart;
window.createAdvancedRadarChart = createAdvancedRadarChart;
window.createHealthScoreChart = createHealthScoreChart;
window.createTrendAnalysisChart = createTrendAnalysisChart;
window.createCategoryBreakdownChart = createCategoryBreakdownChart;
window.createMonthlyComparisonChart = createMonthlyComparisonChart;
window.createExpenseDistributionChart = createExpenseDistributionChart;
window.createSavingsProgressChart = createSavingsProgressChart;

// Initialize on document ready
document.addEventListener('DOMContentLoaded', function() {
    // Clean up charts on page unload
    window.addEventListener('beforeunload', function() {
        ChartManager.destroyAll();
    });
    
    console.log('Enhanced Chart.js utilities loaded successfully');
});
