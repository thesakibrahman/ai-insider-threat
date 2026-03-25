const API_BASE = 'http://localhost:8000/api';

document.addEventListener('DOMContentLoaded', () => {
    const btnSimulate = document.getElementById('btn-simulate');
    const simStatus = document.getElementById('sim-status');
    const tbBody = document.getElementById('anomaly-tbody');
    const modal = document.getElementById('shap-modal');
    const closeBtn = document.querySelector('.close-btn');
    const iframe = document.getElementById('graph-iframe');
    const placeholder = document.getElementById('graph-placeholder');
    const btnRefreshGraph = document.getElementById('btn-refresh-graph');

    // Stats
    const elStats = {
        total: document.getElementById('stat-total'),
        anomalies: document.getElementById('stat-anomalies'),
        precision: document.getElementById('stat-precision'),
        recall: document.getElementById('stat-recall'),
        f1_score: document.getElementById('stat-f1'),
        mttd: document.getElementById('stat-mttd'),
        badge: document.getElementById('badge-count')
    };

    // Load initial state if server is already running and has data
    fetchMetrics();
    fetchAnomalies();

    btnSimulate.addEventListener('click', async () => {
        btnSimulate.disabled = true;
        btnSimulate.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing...';
        simStatus.innerText = 'Generating logs & analyzing...';

        try {
            const res = await fetch(`${API_BASE}/simulate`, { method: 'POST' });
            const data = await res.json();
            
            if (data.status === 'success') {
                updateStats(data.metrics);
                await fetchAnomalies();
                loadGraph();
                simStatus.innerText = 'Pipeline Complete';
            }
        } catch (error) {
            console.error('Error simulating:', error);
            simStatus.innerText = 'Error running simulation';
        } finally {
            btnSimulate.disabled = false;
            btnSimulate.innerHTML = '<i class="fa-solid fa-play"></i> Run Simulation';
        }
    });

    btnRefreshGraph.addEventListener('click', loadGraph);

    async function fetchMetrics() {
        try {
            const res = await fetch(`${API_BASE}/metrics`);
            if (res.ok) {
                const metrics = await res.json();
                updateStats(metrics);
                if (metrics.total_events > 0) loadGraph();
            }
        } catch (e) {
            console.log("No initial metrics");
        }
    }

    async function fetchAnomalies() {
        try {
            const res = await fetch(`${API_BASE}/anomalies`);
            if (res.ok) {
                const data = await res.json();
                renderTable(data.anomalies);
            }
        } catch (e) {
            console.log("No initial anomalies");
        }
    }

    function updateStats(metrics) {
        if (!metrics) return;
        elStats.total.innerText = metrics.total_events || 0;
        elStats.anomalies.innerText = metrics.anomalies_detected || 0;
        elStats.precision.innerText = `${metrics.simulated_precision || 0}%`;
        elStats.recall.innerText = `${metrics.simulated_recall || 0}%`;
        elStats.f1_score.innerText = `${metrics.f1_score || 0}%`;
        elStats.mttd.innerText = `${metrics.mttd || '0s'}`;
        elStats.badge.innerText = `${metrics.anomalies_detected || 0} threats`;
    }

    function renderTable(anomalies) {
        if (!anomalies || anomalies.length === 0) return;
        
        tbBody.innerHTML = '';
        anomalies.forEach(a => {
            const time = new Date(a.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
            
            let riskClass = 'risk-low';
            let riskLabel = 'Low';
            if (a.anomaly_score > 0.8) { riskClass = 'risk-high'; riskLabel = 'High'; }
            else if (a.anomaly_score > 0.6) { riskClass = 'risk-med'; riskLabel = 'Medium'; }

            // Highlight simulated attacks
            const simulatedIcon = a.is_simulated_attack ? `<i class="fa-solid fa-biohazard text-danger" title="Simulated Attack"></i> ` : '';

            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${time}</td>
                <td>
                    <div style="font-weight: 500">${simulatedIcon}${a.user}</div>
                    <div style="font-size: 11px; color: var(--text-secondary)">${a.role}</div>
                </td>
                <td>
                    <div>${a.event_type}</div>
                    <div style="font-size: 11px; color: var(--text-secondary); max-width: 200px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${a.details}</div>
                </td>
                <td class="${riskClass}">${(a.anomaly_score * 100).toFixed(1)}% - ${riskLabel}</td>
                <td>
                    <button class="btn-explain" data-id="${a.log_id}">
                        <i class="fa-solid fa-microscope"></i> Explain XAI
                    </button>
                </td>
            `;
            tbBody.appendChild(tr);
        });

        // Add event listeners to buttons
        document.querySelectorAll('.btn-explain').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const id = e.currentTarget.getAttribute('data-id');
                openExplainer(id);
            });
        });
    }

    function loadGraph() {
        iframe.src = `${API_BASE}/graph?t=${new Date().getTime()}`; // cache buster
        iframe.style.display = 'block';
        placeholder.style.display = 'none';
        
        // Polling to adjust height if needed, handled passively by iframe width=100% height=100%
    }

    async function openExplainer(logId) {
        const containerShap = document.getElementById('shap-chart-container');
        const containerLime = document.getElementById('lime-chart-container');
        containerShap.innerHTML = '<div style="text-align: center; padding: 20px;"><i class="fa-solid fa-circle-notch fa-spin fa-2x"></i></div>';
        if (containerLime) containerLime.innerHTML = '<div style="text-align: center; padding: 20px;"><i class="fa-solid fa-circle-notch fa-spin fa-2x"></i></div>';
        modal.classList.add('show');

        try {
            const res = await fetch(`${API_BASE}/explain/${logId}`);
            if (res.ok) {
                const data = await res.json();
                renderChart(data.explanation.shap, 'shap-chart-container');
                if (containerLime && data.explanation.lime) {
                    renderChart(data.explanation.lime, 'lime-chart-container');
                }
            } else {
                containerShap.innerHTML = '<p style="color:red">Failed to generate explanation</p>';
                if (containerLime) containerLime.innerHTML = '';
            }
        } catch (e) {
            containerShap.innerHTML = `<p style="color:red">Error: ${e.message}</p>`;
            if (containerLime) containerLime.innerHTML = '';
        }
    }

    function renderChart(explanation, containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        if (!explanation || Object.keys(explanation).length === 0) {
            container.innerHTML = '<p>No feature data available.</p>';
            return;
        }

        let html = '';
        // Find max abs value to scale widths
        const maxAbs = Math.max(...Object.values(explanation).map(Math.abs));

        // Select top 8 features to show
        const topFeatures = Object.entries(explanation)
            .sort((a,b) => Math.abs(b[1]) - Math.abs(a[1]))
            .slice(0, 8);

        for (const [feat, val] of topFeatures) {
            if (val === 0) continue; 
            
            const pct = maxAbs > 0 ? (Math.abs(val) / maxAbs) * 100 : 0;
            const color = val > 0 ? 'var(--danger)' : 'var(--accent-color)'; // Red increases risk, Blue decreases
            const icon = val > 0 ? '<i class="fa-solid fa-arrow-up"></i>' : '<i class="fa-solid fa-arrow-down"></i>';

            html += `
                <div class="shap-bar-row">
                    <div class="shap-label" title="${feat}">${feat.replace('feat_', '')}</div>
                    <div class="shap-bar-track">
                        <div class="shap-bar-fill" style="width: ${pct}%; background-color: ${color}"></div>
                    </div>
                    <div class="shap-val" style="color: ${color}">${icon} ${val.toFixed(3)}</div>
                </div>
            `;
        }
        
        // legend
        html += `
            <div style="margin-top: 10px; font-size: 11px; color: var(--text-secondary); display: flex; gap: 10px; justify-content: center;">
                <div><span style="color: var(--danger); font-weight: bold;">■</span> Increases Anomaly Score</div>
                <div><span style="color: var(--accent-color); font-weight: bold;">■</span> Decreases Anomaly Score</div>
            </div>
        `;

        container.innerHTML = html;
        
        // Trigger animations
        setTimeout(() => {
            const fills = container.querySelectorAll('.shap-bar-fill');
            fills.forEach(f => {
                const w = f.style.width;
                f.style.width = '0';
                setTimeout(() => { f.style.width = w; }, 50);
            });
        }, 50);
    }

    closeBtn.addEventListener('click', () => {
        modal.classList.remove('show');
    });

    window.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.classList.remove('show');
        }
    });
});