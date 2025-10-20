// Export Results Function
function exportResults() {
    // Get data from the page
    const results = {
        incident: {
            id: document.querySelector('code').textContent,
            description: document.querySelector('.description-box p').textContent.trim(),
            source: document.querySelectorAll('.info-item strong')[0]?.textContent.trim(),
            reported_at: document.querySelectorAll('.info-item strong')[1]?.textContent.trim()
        },
        analysis: {
            incident_type: document.querySelectorAll('.analysis-value')[0]?.textContent.trim(),
            pattern_match: document.querySelectorAll('.analysis-value')[1]?.textContent.trim(),
            urgency: document.querySelector('.urgency-badge')?.textContent.trim(),
            impact: document.querySelectorAll('.analysis-value')[2]?.textContent.trim(),
            root_cause: document.querySelectorAll('.analysis-value')[3]?.textContent.trim(),
            affected_systems: Array.from(document.querySelectorAll('.system-tag')).map(el => el.textContent.trim())
        },
        resolution_plan: {
            summary: document.querySelector('.summary-box p')?.textContent.trim(),
            steps: Array.from(document.querySelectorAll('.timeline-item')).map((item, index) => ({
                order: index + 1,
                description: item.querySelector('h6')?.textContent.trim(),
                type: item.querySelector('.badge')?.textContent.trim(),
                query: item.querySelector('.query-box code')?.textContent.trim() || null
            }))
        },
        exported_at: new Date().toISOString()
    };
    
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(results, null, 2));
    const downloadElement = document.createElement('a');
    downloadElement.setAttribute("href", dataStr);
    downloadElement.setAttribute("download", `incident_analysis_${Date.now()}.json`);
    document.body.appendChild(downloadElement);
    downloadElement.click();
    downloadElement.remove();
    
    // Show success message
    showToast('Results exported successfully!', 'success');
}

// Toast Notification
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `alert alert-${type} position-fixed top-0 end-0 m-3 shadow-lg`;
    toast.style.zIndex = '9999';
    toast.innerHTML = `
        <i class="fas fa-check-circle me-2"></i>
        ${message}
    `;
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transition = 'opacity 0.5s ease';
        setTimeout(() => toast.remove(), 500);
    }, 3000);
}

// Smooth Scroll Animation
document.addEventListener('DOMContentLoaded', function() {
    const cards = document.querySelectorAll('.results-card');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, {
        threshold: 0.1
    });
    
    cards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(30px)';
        card.style.transition = `all 0.6s ease ${index * 0.1}s`;
        observer.observe(card);
    });
});

// Copy Incident ID to Clipboard
document.addEventListener('DOMContentLoaded', function() {
    const codeElements = document.querySelectorAll('code');
    codeElements.forEach(code => {
        code.style.cursor = 'pointer';
        code.title = 'Click to copy';
        code.addEventListener('click', function() {
            navigator.clipboard.writeText(this.textContent).then(() => {
                showToast('Incident ID copied to clipboard!', 'success');
            });
        });
    });
    
    // Mark step as useful functionality
    setupMarkUsefulButtons();
    
    // Lazy loading functionality
    const loadMoreBtn = document.getElementById('load-more-btn');
    if (loadMoreBtn) {
        loadMoreBtn.addEventListener('click', async function() {
            const incidentId = this.getAttribute('data-incident-id');
            const currentOffset = parseInt(this.getAttribute('data-offset'));
            const totalCount = parseInt(this.getAttribute('data-total'));
            
            // Show loading spinner
            this.disabled = true;
            const spinner = document.getElementById('loading-spinner');
            spinner.classList.remove('d-none');
            
            try {
                const response = await fetch(`/api/load-more-solutions/${incidentId}?offset=${currentOffset}&limit=15`);
                const data = await response.json();
                
                if (response.ok && data.solutions) {
                    // Append new solutions to the container
                    const container = document.getElementById('solutions-container');
                    data.solutions.forEach(solution => {
                        const solutionHtml = createSolutionElement(solution);
                        container.insertAdjacentHTML('beforeend', solutionHtml);
                    });
                    
                    // Re-setup mark useful buttons for new solutions
                    setupMarkUsefulButtons();
                    
                    // Update counter
                    const counter = document.getElementById('solutions-counter');
                    counter.textContent = `Top ${data.loaded_count} of ${data.total_count}`;
                    
                    // Update button state
                    if (data.has_more) {
                        this.setAttribute('data-offset', data.loaded_count);
                        const remaining = data.total_count - data.loaded_count;
                        this.querySelector('.badge').textContent = `${remaining} remaining`;
                    } else {
                        // No more solutions, hide the button
                        document.getElementById('load-more-container').remove();
                        showToast('All solutions loaded!', 'success');
                    }
                } else {
                    showToast('Error loading more solutions', 'danger');
                }
            } catch (error) {
                console.error('Error loading more solutions:', error);
                showToast('Error loading more solutions', 'danger');
            } finally {
                // Hide spinner and re-enable button
                spinner.classList.add('d-none');
                this.disabled = false;
            }
        });
    }
});

// Helper function to create solution HTML element
function createSolutionElement(step) {
    const typeIcons = {
        'Analysis': 'fa-search text-info',
        'Investigation': 'fa-microscope text-warning',
        'Resolution': 'fa-wrench text-success',
        'default': 'fa-check text-primary'
    };
    
    const iconClass = typeIcons[step.type] || typeIcons['default'];
    
    return `
        <div class="timeline-item">
            <div class="timeline-marker">
                <span class="step-number">${step.order}</span>
            </div>
            <div class="timeline-content">
                <div class="timeline-header">
                    <div class="d-flex align-items-center justify-content-between mb-2">
                        <div class="d-flex align-items-center">
                            <i class="fas ${iconClass} step-icon me-2"></i>
                            <span class="badge bg-secondary-subtle text-secondary">${step.type}</span>
                            ${step.source ? `<span class="badge bg-primary-subtle text-primary ms-1" title="Source"><i class="fas fa-database me-1"></i>${step.source}</span>` : ''}
                            ${step.category ? `<span class="badge bg-warning-subtle text-warning ms-1">${step.category}</span>` : ''}
                        </div>
                        <div class="d-flex align-items-center gap-2">
                            <span class="badge bg-info-subtle text-info" title="Usefulness count">
                                <i class="fas fa-thumbs-up me-1"></i>${step.usefulness_count}
                            </span>
                            <button class="btn btn-sm btn-outline-success mark-useful-btn" 
                                    data-step-order="${step.order}"
                                    data-step-description="${step.description}"
                                    ${step.knowledge_base_id ? `data-kb-id="${step.knowledge_base_id}"` : ''}
                                    ${step.training_data_id ? `data-td-id="${step.training_data_id}"` : ''}
                                    data-toggled="false">
                                <i class="fas fa-thumbs-up me-1"></i>Useful
                            </button>
                        </div>
                    </div>
                    ${step.title ? `<h6 class="fw-bold mb-2">${step.title}</h6>` : ''}
                    <div class="text-muted">${step.description}</div>
                </div>
                ${step.query ? `
                    <div class="query-box mt-2">
                        <small class="text-muted d-block mb-1">
                            <i class="fas fa-code me-1"></i>Query:
                        </small>
                        <code class="d-block">${step.query}</code>
                    </div>
                ` : ''}
            </div>
        </div>
    `;
}

// Helper function to setup mark useful buttons
function setupMarkUsefulButtons() {
    const markUsefulButtons = document.querySelectorAll('.mark-useful-btn:not(.processed)');
    markUsefulButtons.forEach(button => {
        button.classList.add('processed'); // Prevent double-binding
        button.addEventListener('click', async function(e) {
            e.preventDefault();
            const toggled = this.getAttribute('data-toggled') === 'true';
            const stepOrder = this.getAttribute('data-step-order');
            const stepDescription = this.getAttribute('data-step-description');
            const stepType = this.getAttribute('data-step-type') || 'Resolution';
            const incidentDescription = this.getAttribute('data-incident-description') || 'Unknown incident';
            const kbId = this.getAttribute('data-kb-id');
            const tdId = this.getAttribute('data-td-id');
            const formData = new FormData();
            formData.append('step_order', stepOrder);
            formData.append('step_description', stepDescription);
            formData.append('step_type', stepType);
            formData.append('incident_description', incidentDescription);
            if (kbId) formData.append('knowledge_base_id', kbId);
            if (tdId) formData.append('training_data_id', tdId);
            try {
                let url, successMsg, errorMsg;
                if (!toggled) {
                    url = '/api/mark-step-useful';
                    successMsg = 'Step marked as useful!';
                    errorMsg = 'Error marking step as useful';
                } else {
                    url = '/api/unmark-step-useful';
                    successMsg = 'Step unmarked as useful!';
                    errorMsg = 'Error unmarking step as useful';
                }
                const response = await fetch(url, {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                if (data.success) {
                    // Update the usefulness count badge
                    const badge = this.closest('.timeline-header').querySelector('.badge.bg-info-subtle');
                    if (badge) {
                        // Try to extract the new count from the response if available
                        if (data.usefulness_count !== undefined) {
                            badge.innerHTML = `<i class="fas fa-thumbs-up me-1"></i>${data.usefulness_count}`;
                        } else {
                            // Fallback: decrement/increment visually
                            let count = parseInt(badge.textContent.trim()) || 0;
                            badge.innerHTML = `<i class="fas fa-thumbs-up me-1"></i>${!toggled ? count + 1 : Math.max(count - 1, 0)}`;
                        }
                    }
                    // Toggle button state
                    if (!toggled) {
                        this.classList.remove('btn-outline-success');
                        this.classList.add('btn-success');
                        this.innerHTML = '<i class="fas fa-check me-1"></i>Marked!';
                        this.setAttribute('data-toggled', 'true');
                    } else {
                        this.classList.remove('btn-success');
                        this.classList.add('btn-outline-success');
                        this.innerHTML = '<i class="fas fa-thumbs-up me-1"></i>Useful';
                        this.setAttribute('data-toggled', 'false');
                    }
                    showToast(successMsg, 'success');
                } else {
                    showToast('Error: ' + (data.error || errorMsg), 'danger');
                }
            } catch (error) {
                console.error(errorMsg, error);
                showToast(errorMsg, 'danger');
            }
        });
    });
}
