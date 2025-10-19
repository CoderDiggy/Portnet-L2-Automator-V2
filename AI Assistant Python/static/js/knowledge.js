let deleteKnowledgeId = null;

// Search and Filter Functionality
document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.getElementById('searchInput');
    const categoryFilter = document.getElementById('categoryFilter');
    const sortBy = document.getElementById('sortBy');
    const knowledgeGrid = document.getElementById('knowledgeGrid');
    const noResults = document.getElementById('noResults');
    const items = document.querySelectorAll('.knowledge-item');
    
    // Populate category filter
    const categories = new Set();
    items.forEach(item => {
        const category = item.dataset.category;
        if (category && category !== 'uncategorized') {
            categories.add(category);
        }
    });
    
    categories.forEach(category => {
        const option = document.createElement('option');
        option.value = category;
        option.textContent = category.charAt(0).toUpperCase() + category.slice(1);
        categoryFilter.appendChild(option);
    });
    
    // Calculate stats
    calculateStats(items);
    
    // Filter function
    function filterItems() {
        const searchTerm = searchInput.value.toLowerCase();
        const selectedCategory = categoryFilter.value;
        let visibleCount = 0;
        
        const itemsArray = Array.from(items);
        
        // Sort items
        if (sortBy.value === 'newest') {
            itemsArray.sort((a, b) => new Date(b.dataset.date) - new Date(a.dataset.date));
        } else if (sortBy.value === 'oldest') {
            itemsArray.sort((a, b) => new Date(a.dataset.date) - new Date(b.dataset.date));
        } else if (sortBy.value === 'title') {
            itemsArray.sort((a, b) => a.dataset.title.localeCompare(b.dataset.title));
        }
        
        // Clear grid and reappend sorted items
        knowledgeGrid.innerHTML = '';
        
        itemsArray.forEach(item => {
            const title = item.dataset.title;
            const category = item.dataset.category;
            const keywords = item.dataset.keywords;
            
            const matchesSearch = searchTerm === '' || 
                                title.includes(searchTerm) || 
                                keywords.includes(searchTerm);
            const matchesCategory = selectedCategory === '' || category === selectedCategory;
            
            if (matchesSearch && matchesCategory) {
                item.style.display = '';
                visibleCount++;
            } else {
                item.style.display = 'none';
            }
            
            knowledgeGrid.appendChild(item);
        });
        
        // Show/hide no results message
        if (visibleCount === 0) {
            noResults.style.display = 'block';
        } else {
            noResults.style.display = 'none';
        }
    }
    
    // Event listeners
    if (searchInput) searchInput.addEventListener('input', filterItems);
    if (categoryFilter) categoryFilter.addEventListener('change', filterItems);
    if (sortBy) sortBy.addEventListener('change', filterItems);
});

// Calculate Statistics
function calculateStats(items) {
    const categories = new Set();
    const keywords = new Set();
    let recentCount = 0;
    const oneMonthAgo = new Date();
    oneMonthAgo.setMonth(oneMonthAgo.getMonth() - 1);
    
    items.forEach(item => {
        // Categories
        const category = item.dataset.category;
        if (category && category !== 'uncategorized') {
            categories.add(category);
        }
        
        // Keywords
        const itemKeywords = item.dataset.keywords.split(',');
        itemKeywords.forEach(kw => {
            if (kw.trim()) keywords.add(kw.trim());
        });
        
        // Recent entries
        const date = new Date(item.dataset.date);
        if (date > oneMonthAgo) {
            recentCount++;
        }
    });
    
    document.getElementById('categoryCount').textContent = categories.size;
    document.getElementById('keywordCount').textContent = keywords.size;
    document.getElementById('recentCount').textContent = recentCount;
}

// View Knowledge Details
function viewKnowledge(id) {
    const modal = new bootstrap.Modal(document.getElementById('viewModal'));
    const modalBody = document.getElementById('viewModalBody');
    
    // Show loading
    modalBody.innerHTML = `
        <div class="text-center py-5">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
        </div>
    `;
    
    modal.show();
    
    // Fetch details (you'll need to implement this endpoint)
    fetch(`/api/knowledge/${id}`)
        .then(response => response.json())
        .then(data => {
            modalBody.innerHTML = `
                <h5 class="fw-bold mb-3">${data.title}</h5>
                ${data.category ? `<span class="badge bg-primary mb-3">${data.category}</span>` : ''}
                <p class="text-muted">${data.content}</p>
                ${data.keywords ? `
                    <div class="mt-3">
                        <h6 class="fw-bold">Keywords:</h6>
                        <div class="d-flex flex-wrap gap-2">
                            ${data.keywords.split(',').map(kw => `<span class="badge bg-light text-dark">${kw.trim()}</span>`).join('')}
                        </div>
                    </div>
                ` : ''}
                <div class="mt-3 pt-3 border-top">
                    <small class="text-muted">
                        <i class="fas fa-calendar me-1"></i> Added: ${new Date(data.created_at).toLocaleString()}<br>
                        <i class="fas fa-source me-1"></i> Source: ${data.source || 'Manual'}
                    </small>
                </div>
            `;
        })
        .catch(error => {
            console.error('Error:', error);
            modalBody.innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-circle me-2"></i>
                    Error loading knowledge entry details
                </div>
            `;
        });
}

// Delete Knowledge Entry
function deleteKnowledge(id, name) {
    deleteKnowledgeId = id;
    document.getElementById('deleteItemName').textContent = name;
    new bootstrap.Modal(document.getElementById('deleteModal')).show();
}

document.getElementById('confirmDeleteBtn').addEventListener('click', function() {
    if (deleteKnowledgeId) {
        this.disabled = true;
        this.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span>Deleting...';
        
        fetch(`/api/knowledge/${deleteKnowledgeId}`, {
            method: 'DELETE',
            headers: {
                'Content-Type': 'application/json',
            }
        })
        .then(response => {
            if (response.ok) {
                showToast('Knowledge entry deleted successfully', 'success');
                setTimeout(() => location.reload(), 1000);
            } else {
                showToast('Error deleting knowledge entry', 'danger');
                this.disabled = false;
                this.innerHTML = '<i class="fas fa-trash me-1"></i>Delete Entry';
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showToast('Error deleting knowledge entry', 'danger');
            this.disabled = false;
            this.innerHTML = '<i class="fas fa-trash me-1"></i>Delete Entry';
        });
    }
});

// Toast Notification
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `alert alert-${type} position-fixed top-0 end-0 m-3 shadow-lg`;
    toast.style.zIndex = '9999';
    toast.innerHTML = `
        <i class="fas fa-${type === 'success' ? 'check-circle' : 'exclamation-circle'} me-2"></i>
        ${message}
    `;
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transition = 'opacity 0.5s ease';
        setTimeout(() => toast.remove(), 500);
    }, 3000);
}
