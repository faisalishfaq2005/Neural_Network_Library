import streamlit as st

st.set_page_config(
    page_title="Neural Network Library",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="auto",
)

# Mark home as active so regression/classification pages know to reset on next visit
st.session_state["active_page"] = "home"

# ── Global styles ────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #0D1117; }
[data-testid="stSidebar"]          { background-color: #161B22; }
.block-container { padding-top: 0rem !important; padding-bottom: 2rem !important; max-width: 1200px; }

/* Hero */
.hero {
    text-align: center;
    padding: 64px 20px 52px;
    background: radial-gradient(ellipse at 50% 0%, rgba(0,217,255,0.09) 0%, transparent 68%);
    border-bottom: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 52px;
}
.hero-badge {
    display: inline-block;
    background: rgba(0,217,255,0.1);
    border: 1px solid rgba(0,217,255,0.3);
    color: #00D9FF;
    font-size: 11px;
    font-family: 'Courier New', monospace;
    letter-spacing: 2.5px;
    padding: 5px 18px;
    border-radius: 20px;
    margin-bottom: 26px;
}
.hero-title {
    font-size: 3.5rem;
    font-weight: 800;
    font-family: Arial, sans-serif;
    background: linear-gradient(135deg, #00D9FF 0%, #D97BFF 50%, #FFB830 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
    margin-bottom: 18px;
}
.hero-sub {
    font-size: 1.1rem;
    color: rgba(200,210,220,0.72);
    font-family: Arial, sans-serif;
    max-width: 660px;
    margin: 0 auto 40px;
    line-height: 1.7;
}
.hero-stats {
    display: flex;
    justify-content: center;
    gap: 56px;
    padding-top: 30px;
    border-top: 1px solid rgba(255,255,255,0.06);
}
.hero-stat { text-align: center; }
.hero-stat-val {
    font-size: 1.9rem; font-weight: 700;
    color: #00D9FF; font-family: 'Courier New', monospace;
}
.hero-stat-lbl {
    font-size: 0.72rem; color: rgba(200,200,200,0.45);
    letter-spacing: 1.5px; text-transform: uppercase; font-family: Arial, sans-serif;
    margin-top: 3px;
}

/* Section header */
.sh { display: flex; align-items: flex-start; gap: 14px; margin: 52px 0 26px; }
.sh-accent { width: 4px; min-height: 36px; border-radius: 2px; margin-top: 3px; flex-shrink: 0; }
.sh-title {
    font-size: 1.45rem; font-weight: 700; color: #E6EDF3;
    font-family: Arial, sans-serif; margin: 0 0 4px;
}
.sh-desc { font-size: 0.88rem; color: rgba(175,185,195,0.6); font-family: Arial, sans-serif; margin: 0; }

/* Cards */
.card {
    border-radius: 13px; padding: 26px 24px; height: 100%;
    position: relative; overflow: hidden;
}
.card::after {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 2px; border-radius: 13px 13px 0 0;
}
.card-c  { background: rgba(0,217,255,0.06);   border: 1px solid rgba(0,217,255,0.2);   }
.card-c::after  { background: linear-gradient(90deg,#00D9FF,transparent); }
.card-p  { background: rgba(217,123,255,0.06); border: 1px solid rgba(217,123,255,0.2); }
.card-p::after  { background: linear-gradient(90deg,#D97BFF,transparent); }
.card-g  { background: rgba(0,255,150,0.06);   border: 1px solid rgba(0,255,150,0.2);   }
.card-g::after  { background: linear-gradient(90deg,#00FF9D,transparent); }
.card-o  { background: rgba(255,184,48,0.06);  border: 1px solid rgba(255,184,48,0.2);  }
.card-o::after  { background: linear-gradient(90deg,#FFB830,transparent); }

.cicon { font-size: 2rem; margin-bottom: 13px; display: block; }
.csub  {
    font-size: 0.72rem; letter-spacing: 1.5px; text-transform: uppercase;
    font-family: 'Courier New', monospace; margin-bottom: 8px;
}
.card-c .csub { color: #00D9FF; }  .card-p .csub { color: #D97BFF; }
.card-g .csub { color: #00FF9D; }  .card-o .csub { color: #FFB830; }
.ctitle {
    font-size: 1rem; font-weight: 700; color: #E6EDF3;
    font-family: Arial, sans-serif; margin-bottom: 10px;
}
.ctext {
    font-size: 0.85rem; color: rgba(185,195,205,0.82);
    font-family: Arial, sans-serif; line-height: 1.68; margin-bottom: 12px;
}
.blist { list-style: none; padding: 0; margin: 0 0 14px; }
.blist li {
    font-size: 0.83rem; color: rgba(185,195,205,0.78);
    font-family: Arial, sans-serif; padding: 3px 0 3px 18px;
    position: relative; line-height: 1.5;
}
.blist li::before { content: '▸'; position: absolute; left: 0; font-size: 0.72rem; }
.card-c .blist li::before { color: #00D9FF; } .card-p .blist li::before { color: #D97BFF; }
.card-g .blist li::before { color: #00FF9D; } .card-o .blist li::before { color: #FFB830; }
.tags { display: flex; flex-wrap: wrap; gap: 6px; }
.tag  { font-size: 0.69rem; padding: 3px 9px; border-radius: 10px; font-family: 'Courier New', monospace; }
.tag-c { background:rgba(0,217,255,0.1);   color:#00D9FF; border:1px solid rgba(0,217,255,0.22); }
.tag-p { background:rgba(217,123,255,0.1); color:#D97BFF; border:1px solid rgba(217,123,255,0.22); }
.tag-g { background:rgba(0,255,150,0.1);   color:#00FF9D; border:1px solid rgba(0,255,150,0.22); }
.tag-o { background:rgba(255,184,48,0.1);  color:#FFB830; border:1px solid rgba(255,184,48,0.22); }

/* Overview box */
.obox {
    background: rgba(255,255,255,0.025); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 13px; padding: 30px 36px; margin-bottom: 4px;
}
.otext {
    font-size: 0.96rem; color: rgba(195,205,215,0.82);
    font-family: Arial, sans-serif; line-height: 1.85; margin: 0;
}
.hl-c { color:#00D9FF; font-weight:600; }
.hl-p { color:#D97BFF; font-weight:600; }
.hl-g { color:#00FF9D; font-weight:600; }
.hl-o { color:#FFB830; font-weight:600; }

/* Workflow */
.wf { display:flex; gap:0; position:relative; margin: 28px 0 8px; }
.wf::before {
    content:''; position:absolute; top:23px; left:5%; right:5%; height:2px;
    background: linear-gradient(90deg,#00D9FF,#82AAFF,#D97BFF,#FFB830,#00FF9D); z-index:0;
}
.step { flex:1; text-align:center; position:relative; z-index:1; padding:0 6px; }
.snum {
    width:46px; height:46px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:0.95rem; font-weight:700; font-family:'Courier New',monospace;
    margin:0 auto 14px; border:2px solid;
}
.step:nth-child(1) .snum { background:rgba(0,217,255,0.14);   border-color:#00D9FF; color:#00D9FF; }
.step:nth-child(2) .snum { background:rgba(130,170,255,0.14); border-color:#82AAFF; color:#82AAFF; }
.step:nth-child(3) .snum { background:rgba(217,123,255,0.14); border-color:#D97BFF; color:#D97BFF; }
.step:nth-child(4) .snum { background:rgba(255,184,48,0.14);  border-color:#FFB830; color:#FFB830; }
.step:nth-child(5) .snum { background:rgba(0,255,150,0.14);   border-color:#00FF9D; color:#00FF9D; }
.stitle { font-size:0.9rem; font-weight:700; color:#E6EDF3; font-family:Arial,sans-serif; margin-bottom:7px; }
.sdesc  { font-size:0.78rem; color:rgba(165,175,185,0.72); font-family:Arial,sans-serif; line-height:1.55; }

/* Divider */
.div { height:1px; background:linear-gradient(90deg,transparent,rgba(255,255,255,0.07),transparent); margin:46px 0; }

/* Navigation buttons */
[data-testid="stButton"] > button {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #E6EDF3 !important; border-radius: 11px !important;
    font-size: 1rem !important; font-family: Arial, sans-serif !important;
    font-weight: 600 !important; padding: 16px 20px !important;
    width: 100% !important; min-height: 84px !important;
    transition: all 0.2s ease !important; line-height: 1.5 !important;
}
[data-testid="stButton"] > button:hover {
    border-color: rgba(0,217,255,0.45) !important;
    background: rgba(0,217,255,0.07) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 20px rgba(0,217,255,0.12) !important;
}

/* Footer */
.footer {
    text-align:center; padding:32px 0 10px;
    border-top:1px solid rgba(255,255,255,0.055); margin-top:48px;
}
.ftext { font-size:0.82rem; color:rgba(150,160,170,0.45); font-family:'Courier New',monospace; }

code {
    background: rgba(255,255,255,0.07); color: #00D9FF;
    padding: 1px 6px; border-radius: 4px;
    font-family: 'Courier New', monospace; font-size: 0.85em;
}
</style>
""", unsafe_allow_html=True)

# ── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">DSA FINAL PROJECT &nbsp;·&nbsp; NEURAL NETWORK LIBRARY</div>
  <h1 class="hero-title">Neural Network Library</h1>
  <p class="hero-sub">
    An interactive platform for building and training neural networks from scratch,
    powered entirely by custom-built data structures — Linked Lists, Stacks, Queues, and Graphs.
  </p>
  <div class="hero-stats">
    <div class="hero-stat">
      <div class="hero-stat-val">4</div>
      <div class="hero-stat-lbl">Data Structures</div>
    </div>
    <div class="hero-stat">
      <div class="hero-stat-val" style="color:#D97BFF">3</div>
      <div class="hero-stat-lbl">Layer Types</div>
    </div>
    <div class="hero-stat">
      <div class="hero-stat-val" style="color:#00FF9D">3</div>
      <div class="hero-stat-lbl">NLP Tasks</div>
    </div>
    <div class="hero-stat">
      <div class="hero-stat-val" style="color:#FFB830">2</div>
      <div class="hero-stat-lbl">Loss Functions</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── OVERVIEW ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="sh">
  <div class="sh-accent" style="background:linear-gradient(180deg,#00D9FF,#D97BFF)"></div>
  <div>
    <p class="sh-title">Project Overview</p>
    <p class="sh-desc">What this application does and what makes it unique</p>
  </div>
</div>
<div class="obox">
  <p class="otext">
    This application serves as an interactive platform for building and training neural networks to solve
    <span class="hl-c">regression</span> and <span class="hl-p">binary classification</span> problems.
    Users can experiment with different model configurations, train the network, and evaluate its performance in real time.<br><br>
    The app also integrates <span class="hl-g">NLP tasks</span>, enabling users to analyze text by performing context analysis,
    extracting keywords, and clustering words based on semantic similarity.<br><br>
    What sets this project apart is its use of <span class="hl-o">custom data structures</span> implemented from scratch.
    A <b>Linked List</b> stores and traverses the neural network layers. A <b>Stack</b> manages error propagation in reverse-layer order
    during backpropagation. A <b>Queue</b> dispatches mini-batches to the training loop. A <b>Graph</b> models word relationships
    for every NLP task. None of these rely on high-level ML libraries — everything is hand-built.
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="div"></div>', unsafe_allow_html=True)

# ── DATA STRUCTURES ───────────────────────────────────────────────────────────
st.markdown("""
<div class="sh">
  <div class="sh-accent" style="background:#00D9FF"></div>
  <div>
    <p class="sh-title">Custom Data Structures</p>
    <p class="sh-desc">The four foundational components that coordinate every operation in the library</p>
  </div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4, gap="small")
with c1:
    st.markdown("""
    <div class="card card-c">
      <span class="cicon">🔗</span>
      <div class="csub">Linked List</div>
      <div class="ctitle">Neural Network Backbone</div>
      <p class="ctext">Each node in the linked list stores one layer — its weight matrix, bias vector, and activation reference. The entire model is a single linked list traversed forward and backward on every iteration.</p>
      <ul class="blist">
        <li>Layers inserted via <code>insert_node</code></li>
        <li>Forward pass: head → tail traversal</li>
        <li>Backward pass: tail → head traversal</li>
        <li>Weight updates after each batch</li>
      </ul>
      <div class="tags">
        <span class="tag tag-c">O(n) traversal</span>
        <span class="tag tag-c">Dynamic depth</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="card card-p">
      <span class="cicon">📚</span>
      <div class="csub">Stack</div>
      <div class="ctitle">Backpropagation Engine</div>
      <p class="ctext">During backpropagation, layer errors are pushed onto a stack in LIFO order. When popped, gradients are applied in exact reverse layer sequence — precisely what the chain rule demands.</p>
      <ul class="blist">
        <li>Errors pushed forward-to-back</li>
        <li>Gradients popped back-to-front (LIFO)</li>
        <li>Enables correct reverse-order updates</li>
        <li>Weights and biases adjusted per pop</li>
      </ul>
      <div class="tags">
        <span class="tag tag-p">LIFO</span>
        <span class="tag tag-p">O(1) push/pop</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="card card-o">
      <span class="cicon">📋</span>
      <div class="csub">Queue</div>
      <div class="ctitle">Batch Processor</div>
      <p class="ctext">The dataset is split into equal mini-batches that are enqueued before each epoch. The training loop dequeues one batch at a time in FIFO order, ensuring sequential and fair processing.</p>
      <ul class="blist">
        <li>Dataset split into configurable batches</li>
        <li>FIFO ensures unbiased sample order</li>
        <li>Queue refilled at the start of each epoch</li>
        <li>Remainder samples handled in final batch</li>
      </ul>
      <div class="tags">
        <span class="tag tag-o">FIFO</span>
        <span class="tag tag-o">Mini-batch SGD</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class="card card-g">
      <span class="cicon">🕸️</span>
      <div class="csub">Graph</div>
      <div class="ctitle">NLP Knowledge Graph</div>
      <p class="ctext">Words are represented as nodes; co-occurrence relationships within a sliding window become weighted edges. This graph is the foundation for all three NLP tasks in the application.</p>
      <ul class="blist">
        <li>Nodes = unique words in the corpus</li>
        <li>Edges = co-occurrence within a window</li>
        <li>Centrality scores → keyword importance</li>
        <li>Community detection → word clusters</li>
      </ul>
      <div class="tags">
        <span class="tag tag-g">Weighted</span>
        <span class="tag tag-g">Undirected</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="div"></div>', unsafe_allow_html=True)

# ── NEURAL NETWORK LAYERS ─────────────────────────────────────────────────────
st.markdown("""
<div class="sh">
  <div class="sh-accent" style="background:#D97BFF"></div>
  <div>
    <p class="sh-title">Neural Network Architecture</p>
    <p class="sh-desc">The three building blocks that make up every trainable model in this library</p>
  </div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3, gap="small")
with c1:
    st.markdown("""
    <div class="card card-p">
      <span class="cicon">⬛</span>
      <div class="csub">Dense Layer</div>
      <div class="ctitle">Fully Connected Layer</div>
      <p class="ctext">The primary computational unit. Every neuron connects to every neuron in the next layer. Stores a weight matrix and a bias vector, both of which are learned during training via gradient descent.</p>
      <ul class="blist">
        <li>Forward: <code>output = W · x + b</code></li>
        <li>Backward: computes ∂L/∂W and ∂L/∂b</li>
        <li>Parameters = <code>inputs × neurons + neurons</code></li>
        <li>First layer input size auto-detected from data</li>
      </ul>
      <div class="tags">
        <span class="tag tag-p">Trainable</span>
        <span class="tag tag-p">W + b</span>
        <span class="tag tag-p">Gradient descent</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="card card-o">
      <span class="cicon">⚡</span>
      <div class="csub">Activation Layer</div>
      <div class="ctitle">Non-linearity Injection</div>
      <p class="ctext">Applies a non-linear function element-wise to the Dense layer's output. Without activation layers, stacking multiple Dense layers is mathematically equivalent to a single linear transform — the network cannot learn complex patterns.</p>
      <ul class="blist">
        <li><b>ReLU</b>: <code>max(0, x)</code> — fast, sparse, avoids vanishing gradients</li>
        <li><b>Sigmoid</b>: <code>1/(1+e⁻ˣ)</code> — squashes output to [0,1]</li>
        <li>Forward: apply function to each element</li>
        <li>Backward: multiply upstream gradient by derivative</li>
      </ul>
      <div class="tags">
        <span class="tag tag-o">ReLU</span>
        <span class="tag tag-o">Sigmoid</span>
        <span class="tag tag-o">0 parameters</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="card card-c">
      <span class="cicon">📉</span>
      <div class="csub">Loss Functions</div>
      <div class="ctitle">Training Signal Generator</div>
      <p class="ctext">Loss functions quantify the gap between the model's prediction and the true label. The gradient of the loss with respect to the output is the signal that drives all weight updates through backpropagation.</p>
      <ul class="blist">
        <li><b>MSE</b> (Regression): mean squared error, penalises large deviations</li>
        <li><b>BCE</b> (Classification): binary cross-entropy for probability outputs</li>
        <li>Loss plotted in real time during every training run</li>
        <li>Lower loss = predictions closer to ground truth</li>
      </ul>
      <div class="tags">
        <span class="tag tag-c">MSE · Regression</span>
        <span class="tag tag-c">BCE · Classification</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="div"></div>', unsafe_allow_html=True)

# ── NLP TASKS ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="sh">
  <div class="sh-accent" style="background:#00FF9D"></div>
  <div>
    <p class="sh-title">NLP Tasks</p>
    <p class="sh-desc">Graph-powered natural language processing — no external NLP libraries, just the custom graph structure</p>
  </div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3, gap="small")
with c1:
    st.markdown("""
    <div class="card card-g">
      <span class="cicon">🔍</span>
      <div class="csub">Context Analysis</div>
      <div class="ctitle">Word Relationship Mapping</div>
      <p class="ctext">Determines the relationships between words within a sentence or document. The graph structure visualises word co-occurrences and helps interpret the meaning of ambiguous words based on their neighbouring context.</p>
      <ul class="blist">
        <li>Sliding window co-occurrence analysis</li>
        <li>Context disambiguation for polysemous words</li>
        <li>Neighbor word frequency ranking</li>
        <li>Graph-based relationship visualisation</li>
      </ul>
      <div class="tags">
        <span class="tag tag-g">Co-occurrence</span>
        <span class="tag tag-g">Graph traversal</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="card card-g">
      <span class="cicon">🏷️</span>
      <div class="csub">Keyword Extraction</div>
      <div class="ctitle">Importance Scoring</div>
      <p class="ctext">Identifies the most significant terms in a body of text. Graph centrality measures determine which words appear in the densest clusters. These central words summarise the document and reveal its main themes.</p>
      <ul class="blist">
        <li>Degree centrality scoring per node</li>
        <li>High-frequency cluster identification</li>
        <li>Keywords ranked by graph importance</li>
        <li>Supports document summarisation workflows</li>
      </ul>
      <div class="tags">
        <span class="tag tag-g">Centrality</span>
        <span class="tag tag-g">Ranking</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="card card-g">
      <span class="cicon">🗂️</span>
      <div class="csub">Word Clustering</div>
      <div class="ctitle">Semantic Grouping</div>
      <p class="ctext">Groups words based on semantic similarity. Graph traversal and community detection algorithms — specifically modularity maximisation — form clusters of semantically related terms, revealing the conceptual structure of the text.</p>
      <ul class="blist">
        <li>Graph community detection algorithm</li>
        <li>Modularity maximisation for cluster quality</li>
        <li>Each cluster = semantically similar terms</li>
        <li>Cluster visualisation on interactive graph</li>
      </ul>
      <div class="tags">
        <span class="tag tag-g">Community detection</span>
        <span class="tag tag-g">Modularity</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="div"></div>', unsafe_allow_html=True)

# ── SUPPORTING FUNCTIONS ──────────────────────────────────────────────────────
st.markdown("""
<div class="sh">
  <div class="sh-accent" style="background:#FFB830"></div>
  <div>
    <p class="sh-title">Supporting Functions</p>
    <p class="sh-desc">Data preparation utilities that ensure clean inputs and efficient training</p>
  </div>
</div>
""", unsafe_allow_html=True)

c1, c2 = st.columns(2, gap="small")
with c1:
    st.markdown("""
    <div class="card card-o">
      <span class="cicon">📐</span>
      <div class="csub">Normalization</div>
      <div class="ctitle">Feature Scaling & Categorical Encoding</div>
      <p class="ctext">
        Scales numeric features to the [0, 1] range using min-max normalisation, and automatically
        label-encodes categorical columns. This prevents high-magnitude features from dominating
        gradient updates and significantly accelerates convergence. The min/max values from training
        are stored so that predictions can be correctly denormalised back to the original scale.
      </p>
      <ul class="blist">
        <li>Auto-detects numeric vs categorical columns</li>
        <li>Min-max scaling: <code>(x − min) / (max − min)</code></li>
        <li>Label encoding: categories → 0, 1, 2 … (sorted)</li>
        <li>Stores metadata for prediction denormalisation</li>
        <li>NaN and Inf values replaced with column mean</li>
      </ul>
      <div class="tags">
        <span class="tag tag-o">Min-Max</span>
        <span class="tag tag-o">Label Encoding</span>
        <span class="tag tag-o">NaN handling</span>
        <span class="tag tag-o">Reversible</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="card card-o">
      <span class="cicon">📦</span>
      <div class="csub">Batching</div>
      <div class="ctitle">Mini-Batch Data Management via Queue</div>
      <p class="ctext">
        Splits the full dataset into equal-sized mini-batches that are loaded into the Queue data structure
        at the start of each epoch. The training loop dequeues one batch at a time. Mini-batch gradient
        descent strikes a balance between the noisy updates of SGD and the expensive full-dataset
        gradient of batch gradient descent, improving both speed and generalisation.
      </p>
      <ul class="blist">
        <li>Configurable batch size (default: 32)</li>
        <li>Queue-based FIFO batch delivery per epoch</li>
        <li>Feature array and label array kept in sync</li>
        <li>Final batch handles any remaining samples</li>
        <li>Queue refilled fresh at the start of each epoch</li>
      </ul>
      <div class="tags">
        <span class="tag tag-o">Mini-batch</span>
        <span class="tag tag-o">Queue-based</span>
        <span class="tag tag-o">Configurable</span>
        <span class="tag tag-o">Per-epoch reset</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="div"></div>', unsafe_allow_html=True)

# ── WORKFLOW ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="sh">
  <div class="sh-accent" style="background:linear-gradient(180deg,#00D9FF,#00FF9D)"></div>
  <div>
    <p class="sh-title">End-to-End Workflow</p>
    <p class="sh-desc">From raw data to trained model and real-time predictions — five steps</p>
  </div>
</div>
<div class="wf">
  <div class="step">
    <div class="snum">01</div>
    <div class="stitle">Upload Data</div>
    <div class="sdesc">Upload a CSV file. The app auto-detects column types (numeric vs categorical) and shows a data preview instantly.</div>
  </div>
  <div class="step">
    <div class="snum">02</div>
    <div class="stitle">Normalise</div>
    <div class="sdesc">Enable normalisation to apply min-max scaling to numeric columns and label-encode categoricals for stable training.</div>
  </div>
  <div class="step">
    <div class="snum">03</div>
    <div class="stitle">Build Network</div>
    <div class="sdesc">Add Dense and Activation layers. Each layer is inserted into the linked list. The architecture is visualised live.</div>
  </div>
  <div class="step">
    <div class="snum">04</div>
    <div class="stitle">Train</div>
    <div class="sdesc">Set epochs, learning rate, and batch size. The queue dispatches batches; the stack propagates gradients backward. Loss updates in real time.</div>
  </div>
  <div class="step">
    <div class="snum">05</div>
    <div class="stitle">Predict</div>
    <div class="sdesc">Enter new values — dropdowns for categorical columns, number inputs for numeric. The trained network returns an instant prediction.</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="div"></div>', unsafe_allow_html=True)

# ── NAVIGATION ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="sh">
  <div class="sh-accent" style="background:linear-gradient(180deg,#D97BFF,#00D9FF)"></div>
  <div>
    <p class="sh-title">Launch a Module</p>
    <p class="sh-desc">Each module starts fresh — upload your dataset and build your network from scratch</p>
  </div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3, gap="small")
with c1:
    if st.button("📈  Regression\nPredict continuous numeric values", key="nav_reg"):
        st.switch_page("pages/regressionpage.py")
with c2:
    if st.button("🎯  Classification\nBinary class prediction with confidence", key="nav_clf"):
        st.switch_page("pages/classificationpage.py")
with c3:
    if st.button("💬  NLP Tasks\nContext analysis, keywords & clustering", key="nav_nlp"):
        st.switch_page("pages/nlpPage.py")

# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  <p class="ftext">Neural Network Library &nbsp;·&nbsp; DSA Final Project &nbsp;·&nbsp; Built with Streamlit + Custom Data Structures</p>
</div>
""", unsafe_allow_html=True)
