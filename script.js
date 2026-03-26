const CONFIG = {
    endpoint: '/generate',
    typeSpeed: 3,
    errorColor: '#EF4444',
    successColor: '#10B981',
    primaryColor: '#6366f1'
};

const dom = {
    generateBtn: document.getElementById('generateBtn'),
    topic: document.getElementById('topic'),
    language: document.getElementById('language'),
    languageGroup: document.getElementById('language-group'),
    
    // New Report DOM
    reportContainer: document.getElementById('report-container'),
    loadingContainer: document.getElementById('loading-container'),
    loadingText: document.getElementById('loading-text'),
    
    extSources: document.getElementById('ext-sources'),
    knowledgeBase: document.getElementById('knowledge-base'),
    generatedContent: document.getElementById('generated-content'),
    
    metricRel: document.getElementById('metric-rel'),
    metricCoh: document.getElementById('metric-coh'),
    metricSim: document.getElementById('metric-sim'),
    metricNli: document.getElementById('metric-nli'),
    metricTech: document.getElementById('metric-tech'),
    metricComp: document.getElementById('metric-comp'),
    metricAcc: document.getElementById('metric-acc'),
    metricEval: document.getElementById('metric-eval'),

    copyBtn: document.getElementById('copyBtn'),
    copyLabel: document.getElementById('copyLabel'),
    sidebar: document.getElementById('sidebar'),
    toggleSidebar: document.getElementById('toggleSidebar'),
    openSidebar: document.getElementById('openSidebar')
};

// Sidebar Toggle Logic
dom.toggleSidebar.addEventListener('click', () => {
    dom.sidebar.classList.add('hidden');
    dom.openSidebar.style.display = 'grid';
});

dom.openSidebar.addEventListener('click', () => {
    dom.sidebar.classList.remove('hidden');
    dom.openSidebar.style.display = 'none';
});

function getSelectedMode() {
    const checked = document.querySelector('input[name="mode"]:checked');
    return checked ? checked.value : 'text';
}

function updateModeUI() {
    if (getSelectedMode() === 'code') {
        dom.languageGroup.style.display = 'flex';
    } else {
        dom.languageGroup.style.display = 'none';
    }
}
document.querySelectorAll('input[name="mode"]').forEach(r => r.addEventListener('change', updateModeUI));
updateModeUI();

dom.generateBtn.addEventListener('click', async () => {
    const topic = dom.topic.value.trim();
    if (!topic) {
        notifyError('Subject input required');
        return;
    }

    resetUIState();
    setProcessingStyle();

    try {
        const response = await fetch(CONFIG.endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                mode: getSelectedMode(),
                topic: topic,
                language: dom.language.value
            })
        });

        const data = await response.json();

        if (data.error) {
            handleSystemError(data.error);
            return;
        }

        handleSystemSuccess(data);

    } catch (err) {
        handleSystemError(err.message);
    }
});

function resetUIState() {
    dom.reportContainer.style.display = 'none';
    dom.loadingContainer.style.display = 'flex';
    dom.loadingText.textContent = 'Crawling knowledge base and running semantic evaluation...';
    
    // Clear old data
    dom.extSources.innerHTML = '';
    dom.knowledgeBase.textContent = '';
    dom.generatedContent.textContent = '';
    
    dom.metricSim.textContent = '0';
    dom.metricNli.textContent = '0';
    dom.metricTech.textContent = '0';
    dom.metricComp.textContent = '0';
    dom.metricAcc.textContent = '0';
    
    const dLabel = document.getElementById('domain-trust-label');
    const dScore = document.getElementById('domain-trust-score');
    if(dLabel) dLabel.textContent = "Source Trust Score";
    if(dScore) dScore.textContent = "--";
    
    const t1 = document.getElementById('t1-cnt');
    const t2 = document.getElementById('t2-cnt');
    const t3 = document.getElementById('t3-cnt');
    if(t1) t1.textContent = "0";
    if(t2) t2.textContent = "0";
    if(t3) t3.textContent = "0";
    dom.metricEval.textContent = '...';
}

function setProcessingStyle() {
    dom.generateBtn.disabled = true;
    dom.generateBtn.style.opacity = '0.5';
}

function handleSystemSuccess(data) {
    dom.generateBtn.disabled = false;
    dom.generateBtn.style.opacity = '1';
    
    dom.loadingContainer.style.display = 'none';
    dom.reportContainer.style.display = 'block';

    // Populate Report Fields
    if (data.external_sources && data.external_sources.length > 0) {
        data.external_sources.forEach((link, idx) => {
            const a = document.createElement('a');
            a.href = link;
            a.target = '_blank';
            a.textContent = `${idx + 1}. ${link}`;
            dom.extSources.appendChild(a);
        });
    } else {
        dom.extSources.textContent = "No external sources considered.";
    }

    dom.knowledgeBase.textContent = data.knowledge_base || "N/A";

    const currentMode = getSelectedMode();
    const knowledgeLabel = dom.knowledgeBase.previousElementSibling;
    const generatedLabel = dom.generatedContent.previousElementSibling;
    if (currentMode === 'code') {
        if (knowledgeLabel) knowledgeLabel.textContent = "💻 EXTRACTED SOURCE CODE";
        if (generatedLabel) generatedLabel.textContent = "🤖 GENERATED CODE";
    } else {
        if (knowledgeLabel) knowledgeLabel.textContent = "🧠 EXTRACTED KNOWLEDGE BASE";
        if (generatedLabel) generatedLabel.textContent = "🤖 GENERATED CONTENT";
    }

    // Metrics
    if (data.evaluation && data.evaluation.metrics) {
        const mGrid = document.querySelector('.metrics-grid');
        mGrid.innerHTML = '';
        
        const labelMap = {
            // Code Mode — 5 metrics
            "source_trust":            "Source Trust Score",
            "functional_accuracy":     "Functional Accuracy (Python)",
            "semantic_similarity":     "Semantic Similarity (CodeBERT)",
            "coverage_score":          "Coverage Score",
            "structural_similarity":   "Structural Similarity (BLEU)",
            // Text Mode — 6 metrics
            "similarity":  "Semantic Similarity Score",
            "nli":         "NLI Entailment Score",
            "relevance":   "Semantic Alignment (Relevance)",
            "coherence":   "Logical Coherence Score",
            "technical":   "Technical Term Retention",
            "compression": "Content Compression Score"
        };

        for (const [key, value] of Object.entries(data.evaluation.metrics)) {
            const row = document.createElement('div');
            row.className = 'metric-row';
            row.innerHTML = `<span>${labelMap[key] || key}</span><span>: <span style="font-weight: bold;">${value}</span></span>`;
            mGrid.appendChild(row);
        }
        
        if (data.domain_weight !== undefined) {
            let activeTiers = [];
            if(data.tier_counts && data.tier_counts.tier1 > 0) activeTiers.push("Tier 1");
            if(data.tier_counts && data.tier_counts.tier2 > 0) activeTiers.push("Tier 2");
            if(data.tier_counts && data.tier_counts.tier3 > 0) activeTiers.push("Tier 3");
            const activeStr = activeTiers.length > 0 ? ` (${activeTiers.join(', ')})` : "";
            
            const dr = document.createElement('div');
            dr.className = 'metric-row';
            dr.innerHTML = `<span id="domain-trust-label" style="font-weight: 600;">Source Trust Score${activeStr}</span><span style="font-weight: bold; color: var(--clr-primary);">: <span id="domain-trust-score">${data.domain_weight.toFixed(2)}</span></span>`;
            mGrid.appendChild(dr);
        }
        // Final Assessment assignments
        dom.metricAcc.textContent = data.evaluation.accuracy_percentage !== undefined ? data.evaluation.accuracy_percentage : "--";
        dom.metricEval.textContent = data.evaluation.evaluation_score;
    }
    
    // Tier Counts Update
    if (data.tier_counts) {
        document.getElementById('t1-cnt').textContent = data.tier_counts.tier1;
        document.getElementById('t2-cnt').textContent = data.tier_counts.tier2;
        document.getElementById('t3-cnt').textContent = data.tier_counts.tier3;
    }

    // Typewriter effect for Generated Content
    typewriterEffect(data.generated_content || "Error generating content", dom.generatedContent, () => {
    });
}

// Copy button handler
dom.copyBtn.addEventListener('click', () => {
    const rawText = dom.generatedContent.textContent;
    navigator.clipboard.writeText(rawText).then(() => {
        dom.copyLabel.textContent = 'Copied!';
        setTimeout(() => { dom.copyLabel.textContent = 'Copy Output'; }, 2000);
    }).catch(() => {
        // Fallback
        const sel = window.getSelection();
        const range = document.createRange();
        range.selectNodeContents(dom.generatedContent);
        sel.removeAllRanges();
        sel.addRange(range);
        document.execCommand('copy');
        sel.removeAllRanges();
        dom.copyLabel.textContent = 'Copied!';
        setTimeout(() => { dom.copyLabel.textContent = 'Copy Output'; }, 2000);
    });
});

window.toggleTier = function(tier) {
    const content = document.getElementById(tier + '-content');
    const arrow = document.getElementById(tier + '-arrow');
    if (content.style.display === 'none') {
        content.style.display = 'block';
        arrow.textContent = '▲';
    } else {
        content.style.display = 'none';
        arrow.textContent = '▼';
    }
};

function handleSystemError(error) {
    dom.generateBtn.disabled = false;
    dom.generateBtn.style.opacity = '1';
    
    dom.loadingContainer.style.display = 'none';
    dom.reportContainer.style.display = 'block';
    dom.generatedContent.textContent = `SYSTEM ERROR: ${error}`;
    dom.knowledgeBase.textContent = "Failed";
}

function notifyError(msg) {
    dom.topic.style.borderColor = CONFIG.errorColor;
    setTimeout(() => dom.topic.style.borderColor = '', 2000);
}

function typewriterEffect(text, element, callback) {
    element.textContent = '';
    let i = 0;

    function type() {
        if (i < text.length) {
            let chunkSize = Math.min(3, text.length - i);
            element.textContent += text.substring(i, i + chunkSize);
            i += chunkSize;
            setTimeout(type, CONFIG.typeSpeed);
        } else if (callback) {
            callback();
        }
    }
    type();
}
