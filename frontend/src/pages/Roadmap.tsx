import React, { useState, useCallback, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

/* ═══════════════════════════════════════════════════════
   ML ROADMAP DATA — based on roadmap.sh/machine-learning
   ═══════════════════════════════════════════════════════ */

interface SubBranch {
    id: string;
    title: string;
    icon: string;
}

interface RoadmapNode {
    id: string;
    title: string;
    description: string;
    icon: string;
    children?: SubBranch[];
}

interface RoadmapCategory {
    id: string;
    title: string;
    icon: string;
    color: string;
    nodes: RoadmapNode[];
}

const roadmapData: RoadmapCategory[] = [
    {
        id: 'fundamentals',
        title: 'Mathematics & Fundamentals',
        icon: '🧮',
        color: '#6366f1',
        nodes: [
            {
                id: 'linear-algebra',
                title: 'Linear Algebra',
                icon: '📐',
                description:
                    'The backbone of ML. Understand vectors, matrices, eigenvalues, and decompositions. Every neural network is a chain of matrix multiplications — linear algebra is how you speak the machine\'s language.',
                children: [
                    { id: 'vectors-matrices', title: 'Vectors & Matrices', icon: '🔢' },
                    { id: 'eigenvalues', title: 'Eigenvalues & Eigenvectors', icon: '🎯' },
                    { id: 'svd', title: 'SVD Decomposition', icon: '🔀' },
                    { id: 'matrix-ops', title: 'Matrix Operations', icon: '✖️' },
                ],
            },
            {
                id: 'calculus',
                title: 'Calculus',
                icon: '📈',
                description:
                    'Derivatives and gradients power optimization. Backpropagation is just the chain rule applied recursively. Multivariate calculus lets you navigate high-dimensional loss landscapes.',
                children: [
                    { id: 'derivatives', title: 'Derivatives & Gradients', icon: '📉' },
                    { id: 'chain-rule', title: 'Chain Rule', icon: '🔗' },
                    { id: 'partial-derivatives', title: 'Partial Derivatives', icon: '∂' },
                    { id: 'integrals', title: 'Integrals', icon: '∫' },
                ],
            },
            {
                id: 'probability-stats',
                title: 'Probability & Statistics',
                icon: '🎲',
                description:
                    'ML is applied statistics. Bayes\' theorem, distributions, hypothesis testing, and maximum likelihood estimation form the probabilistic foundation every model rests on.',
                children: [
                    { id: 'bayes-theorem', title: 'Bayes\' Theorem', icon: '📊' },
                    { id: 'distributions', title: 'Distributions', icon: '📈' },
                    { id: 'hypothesis-testing', title: 'Hypothesis Testing', icon: '🧪' },
                    { id: 'mle', title: 'Maximum Likelihood', icon: '🎯' },
                ],
            },
            {
                id: 'python',
                title: 'Python Programming',
                icon: '🐍',
                description:
                    'The lingua franca of ML. Master NumPy, Pandas, and Matplotlib. Understand vectorized operations, broadcasting, and efficient data manipulation — these are your daily tools.',
                children: [
                    { id: 'numpy', title: 'NumPy', icon: '🔢' },
                    { id: 'pandas', title: 'Pandas', icon: '🐼' },
                    { id: 'matplotlib', title: 'Matplotlib / Seaborn', icon: '📊' },
                    { id: 'jupyter', title: 'Jupyter Notebooks', icon: '📓' },
                ],
            },
            {
                id: 'libraries',
                title: 'ML Libraries',
                icon: '📚',
                description:
                    'Scikit-learn for classical ML, TensorFlow and PyTorch for deep learning. Learn their APIs, data pipelines, and how to leverage GPU acceleration for training.',
                children: [
                    { id: 'sklearn', title: 'Scikit-learn', icon: '🔧' },
                    { id: 'tensorflow', title: 'TensorFlow', icon: '🧠' },
                    { id: 'pytorch', title: 'PyTorch', icon: '🔥' },
                    { id: 'keras', title: 'Keras', icon: '⚡' },
                ],
            },
        ],
    },
    {
        id: 'data-preprocessing',
        title: 'Data Preprocessing',
        icon: '🔧',
        color: '#8b5cf6',
        nodes: [
            {
                id: 'data-collection',
                title: 'Data Collection',
                icon: '📥',
                description:
                    'Garbage in, garbage out. Learn web scraping, API integration, database queries, and dataset curation. Understand data licensing, ethics, and building representative datasets.',
                children: [
                    { id: 'web-scraping', title: 'Web Scraping', icon: '🕷️' },
                    { id: 'api-integration', title: 'API Integration', icon: '🔌' },
                    { id: 'databases', title: 'Database Queries', icon: '🗄️' },
                    { id: 'labeling', title: 'Data Labeling', icon: '🏷️' },
                ],
            },
            {
                id: 'data-cleaning',
                title: 'Data Cleaning',
                icon: '🧹',
                description:
                    'Handle missing values, remove duplicates, fix inconsistencies, and detect outliers. Data scientists spend 80% of their time here — master it and you\'re ahead of the game.',
                children: [
                    { id: 'missing-values', title: 'Missing Values', icon: '❓' },
                    { id: 'outliers', title: 'Outlier Detection', icon: '📍' },
                    { id: 'duplicates', title: 'Deduplication', icon: '♻️' },
                    { id: 'normalization', title: 'Data Normalization', icon: '📏' },
                ],
            },
            {
                id: 'feature-engineering',
                title: 'Feature Engineering',
                icon: '⚙️',
                description:
                    'The art of creating informative features from raw data. Polynomial features, interaction terms, binning, encoding categoricals. This is where domain knowledge becomes your superpower.',
                children: [
                    { id: 'encoding', title: 'Categorical Encoding', icon: '🔤' },
                    { id: 'polynomial', title: 'Polynomial Features', icon: '📐' },
                    { id: 'binning', title: 'Binning / Bucketing', icon: '📦' },
                    { id: 'interaction', title: 'Interaction Terms', icon: '🤝' },
                ],
            },
            {
                id: 'feature-scaling',
                title: 'Feature Scaling & Selection',
                icon: '⚖️',
                description:
                    'Normalization, standardization, and min-max scaling ensure features contribute equally. Feature selection with correlation analysis, mutual information, and recursive elimination reduces dimensionality.',
                children: [
                    { id: 'min-max', title: 'Min-Max Scaling', icon: '📊' },
                    { id: 'standardization', title: 'Standardization', icon: '📏' },
                    { id: 'correlation', title: 'Correlation Analysis', icon: '🔗' },
                    { id: 'rfe', title: 'Recursive Elimination', icon: '🔁' },
                ],
            },
        ],
    },
    {
        id: 'supervised-learning',
        title: 'Supervised Learning',
        icon: '🎯',
        color: '#a855f7',
        nodes: [
            {
                id: 'linear-regression',
                title: 'Linear Regression',
                icon: '📉',
                description:
                    'The "Hello World" of ML. Fit a line to data using least squares. Understand cost functions, gradient descent, and regularization (Ridge, Lasso). Simple but foundational.',
                children: [
                    { id: 'least-squares', title: 'Least Squares', icon: '📐' },
                    { id: 'ridge', title: 'Ridge (L2)', icon: '🏔️' },
                    { id: 'lasso', title: 'Lasso (L1)', icon: '🎯' },
                    { id: 'gradient-descent', title: 'Gradient Descent', icon: '⬇️' },
                ],
            },
            {
                id: 'logistic-regression',
                title: 'Logistic Regression',
                icon: '🔀',
                description:
                    'Despite the name, it\'s for classification. The sigmoid function maps predictions to probabilities. Understand decision boundaries, log-loss, and multi-class extensions (softmax).',
                children: [
                    { id: 'sigmoid', title: 'Sigmoid Function', icon: '〰️' },
                    { id: 'log-loss', title: 'Log Loss', icon: '📉' },
                    { id: 'softmax', title: 'Softmax (Multi-class)', icon: '🔢' },
                ],
            },
            {
                id: 'decision-trees',
                title: 'Decision Trees',
                icon: '🌳',
                description:
                    'Intuitive, interpretable models that split data using information gain or Gini impurity. Prone to overfitting but the foundation for powerful ensemble methods like Random Forests.',
                children: [
                    { id: 'info-gain', title: 'Information Gain', icon: 'ℹ️' },
                    { id: 'gini', title: 'Gini Impurity', icon: '📊' },
                    { id: 'pruning', title: 'Pruning', icon: '✂️' },
                    { id: 'cart', title: 'CART Algorithm', icon: '🧮' },
                ],
            },
            {
                id: 'random-forests',
                title: 'Random Forests',
                icon: '🌲',
                description:
                    'An ensemble of decorrelated decision trees. Bagging + random feature subsets = reduced variance. Robust, handles mixed data types, and provides feature importance rankings.',
            },
            {
                id: 'svm',
                title: 'Support Vector Machines',
                icon: '✂️',
                description:
                    'Find the maximum-margin hyperplane separating classes. The kernel trick projects data into higher dimensions where linear separation is possible. Elegant math, powerful results.',
                children: [
                    { id: 'kernel-trick', title: 'Kernel Trick', icon: '🔮' },
                    { id: 'margin', title: 'Maximum Margin', icon: '↔️' },
                    { id: 'rbf', title: 'RBF Kernel', icon: '🎯' },
                    { id: 'svm-regression', title: 'SVR', icon: '📈' },
                ],
            },
            {
                id: 'knn',
                title: 'K-Nearest Neighbors',
                icon: '📍',
                description:
                    'The simplest classifier: predict based on majority vote of K closest training points. No training phase, but expensive at prediction time. Choice of K and distance metric matters.',
            },
            {
                id: 'naive-bayes',
                title: 'Naive Bayes',
                icon: '📊',
                description:
                    'Fast, probabilistic classifier assuming feature independence. Despite the "naive" assumption, works surprisingly well for text classification, spam filtering, and sentiment analysis.',
                children: [
                    { id: 'gaussian-nb', title: 'Gaussian NB', icon: '📈' },
                    { id: 'multinomial-nb', title: 'Multinomial NB', icon: '📝' },
                    { id: 'bernoulli-nb', title: 'Bernoulli NB', icon: '🔢' },
                ],
            },
        ],
    },
    {
        id: 'unsupervised-learning',
        title: 'Unsupervised Learning',
        icon: '🔍',
        color: '#06b6d4',
        nodes: [
            {
                id: 'k-means',
                title: 'K-Means Clustering',
                icon: '🎯',
                description:
                    'Partition data into K clusters by minimizing within-cluster variance. Simple, fast, but sensitive to initialization and assumes spherical clusters. The elbow method helps choose K.',
                children: [
                    { id: 'elbow-method', title: 'Elbow Method', icon: '📐' },
                    { id: 'k-means-pp', title: 'K-Means++', icon: '➕' },
                    { id: 'silhouette', title: 'Silhouette Score', icon: '📊' },
                ],
            },
            {
                id: 'hierarchical-clustering',
                title: 'Hierarchical Clustering',
                icon: '🏔️',
                description:
                    'Build a tree (dendrogram) of cluster merges. Agglomerative (bottom-up) or divisive (top-down). No need to pre-specify K. Great for understanding data structure at multiple scales.',
                children: [
                    { id: 'agglomerative', title: 'Agglomerative', icon: '⬆️' },
                    { id: 'divisive', title: 'Divisive', icon: '⬇️' },
                    { id: 'dendrogram', title: 'Dendrograms', icon: '🌳' },
                ],
            },
            {
                id: 'pca',
                title: 'PCA (Dimensionality Reduction)',
                icon: '🔬',
                description:
                    'Principal Component Analysis finds directions of maximum variance. Reduce dimensions while preserving information. Essential for visualization, noise reduction, and defeating the curse of dimensionality.',
                children: [
                    { id: 'variance-explained', title: 'Explained Variance', icon: '📊' },
                    { id: 'scree-plot', title: 'Scree Plot', icon: '📉' },
                    { id: 'tsne', title: 't-SNE', icon: '🗺️' },
                    { id: 'umap', title: 'UMAP', icon: '🌐' },
                ],
            },
            {
                id: 'dbscan',
                title: 'DBSCAN',
                icon: '🗺️',
                description:
                    'Density-based clustering that finds arbitrary-shaped clusters and automatically identifies outliers. Unlike K-Means, doesn\'t require specifying the number of clusters. Sensitive to epsilon and min-points.',
            },
        ],
    },
    {
        id: 'model-evaluation',
        title: 'Model Evaluation & Validation',
        icon: '✅',
        color: '#22c55e',
        nodes: [
            {
                id: 'cross-validation',
                title: 'Cross Validation',
                icon: '🔄',
                description:
                    'K-Fold CV, Stratified CV, Leave-One-Out. Split data multiple ways to get robust performance estimates. Prevents overfitting to a single train/test split and gives confidence intervals.',
                children: [
                    { id: 'k-fold', title: 'K-Fold CV', icon: '🔢' },
                    { id: 'stratified', title: 'Stratified CV', icon: '📊' },
                    { id: 'loo', title: 'Leave-One-Out', icon: '1️⃣' },
                ],
            },
            {
                id: 'confusion-matrix',
                title: 'Confusion Matrix & Metrics',
                icon: '📋',
                description:
                    'Precision, Recall, F1-Score, Accuracy. Understand TP, FP, TN, FN. Choose the right metric for your problem — accuracy is misleading for imbalanced datasets. F1 balances precision and recall.',
                children: [
                    { id: 'precision', title: 'Precision', icon: '🎯' },
                    { id: 'recall', title: 'Recall', icon: '🔎' },
                    { id: 'f1-score', title: 'F1 Score', icon: '⚖️' },
                    { id: 'accuracy', title: 'Accuracy', icon: '✅' },
                ],
            },
            {
                id: 'roc-auc',
                title: 'ROC Curve & AUC',
                icon: '📊',
                description:
                    'The ROC curve plots True Positive Rate vs False Positive Rate at all thresholds. AUC (Area Under Curve) gives a single number summarizing classifier performance. Higher = better.',
            },
            {
                id: 'bias-variance',
                title: 'Bias-Variance Tradeoff',
                icon: '⚖️',
                description:
                    'The fundamental tension in ML. High bias = underfitting, high variance = overfitting. Model complexity controls the tradeoff. Regularization, ensembles, and more data help find the sweet spot.',
                children: [
                    { id: 'underfitting', title: 'Underfitting', icon: '📉' },
                    { id: 'overfitting', title: 'Overfitting', icon: '📈' },
                    { id: 'regularization', title: 'Regularization', icon: '🔧' },
                ],
            },
        ],
    },
    {
        id: 'ensemble-methods',
        title: 'Ensemble Methods',
        icon: '🏗️',
        color: '#f97316',
        nodes: [
            {
                id: 'bagging',
                title: 'Bagging',
                icon: '👜',
                description:
                    'Bootstrap Aggregating: train multiple models on random subsets of data, then average predictions. Reduces variance without increasing bias. Random Forests are bagging\'s crown jewel.',
                children: [
                    { id: 'bootstrap', title: 'Bootstrap Sampling', icon: '🔄' },
                    { id: 'aggregation', title: 'Aggregation', icon: '📦' },
                    { id: 'oob-score', title: 'Out-of-Bag Score', icon: '📊' },
                ],
            },
            {
                id: 'boosting',
                title: 'Boosting (XGBoost, AdaBoost)',
                icon: '🚀',
                description:
                    'Sequentially train weak learners, each focusing on previous errors. XGBoost dominates Kaggle and industry. Gradient boosting minimizes loss via functional gradient descent. The king of tabular data.',
                children: [
                    { id: 'adaboost', title: 'AdaBoost', icon: '➕' },
                    { id: 'xgboost', title: 'XGBoost', icon: '⚡' },
                    { id: 'lightgbm', title: 'LightGBM', icon: '💡' },
                    { id: 'catboost', title: 'CatBoost', icon: '🐱' },
                ],
            },
            {
                id: 'stacking',
                title: 'Stacking',
                icon: '📚',
                description:
                    'Train diverse base models, then use a meta-learner to combine their predictions. Leverages each model\'s strengths. More complex but can squeeze out extra performance in competitions.',
                children: [
                    { id: 'meta-learner', title: 'Meta-Learner', icon: '🧠' },
                    { id: 'blending', title: 'Blending', icon: '🔀' },
                ],
            },
        ],
    },
    {
        id: 'deep-learning',
        title: 'Deep Learning',
        icon: '🧠',
        color: '#ec4899',
        nodes: [
            {
                id: 'neural-networks',
                title: 'Neural Networks',
                icon: '🕸️',
                description:
                    'Universal function approximators. Layers of interconnected neurons with non-linear activations (ReLU, sigmoid). Trained via backpropagation and gradient descent. The foundation of deep learning.',
                children: [
                    { id: 'perceptron', title: 'Perceptron', icon: '⚡' },
                    { id: 'activations', title: 'Activation Functions', icon: '〰️' },
                    { id: 'backprop', title: 'Backpropagation', icon: '🔄' },
                    { id: 'optimizers', title: 'Optimizers (Adam, SGD)', icon: '⬇️' },
                    { id: 'dropout', title: 'Dropout / BatchNorm', icon: '🎲' },
                ],
            },
            {
                id: 'cnn',
                title: 'CNNs (Computer Vision)',
                icon: '👁️',
                description:
                    'Convolutional Neural Networks exploit spatial hierarchies in images. Convolutional layers + pooling = translation-invariant feature detection. Powers image classification, detection, and segmentation.',
                children: [
                    { id: 'conv-layers', title: 'Convolutional Layers', icon: '🔲' },
                    { id: 'pooling', title: 'Pooling Layers', icon: '📦' },
                    { id: 'resnet', title: 'ResNet / VGG', icon: '🏛️' },
                    { id: 'object-detection', title: 'Object Detection', icon: '🔍' },
                    { id: 'segmentation', title: 'Segmentation', icon: '🖼️' },
                ],
            },
            {
                id: 'rnn-lstm',
                title: 'RNNs & LSTMs',
                icon: '🔁',
                description:
                    'Recurrent networks process sequences by maintaining hidden state. LSTMs solve the vanishing gradient problem with gated cells. Used for time series, NLP, and speech — though Transformers have largely taken over.',
                children: [
                    { id: 'vanilla-rnn', title: 'Vanilla RNN', icon: '🔄' },
                    { id: 'lstm-gates', title: 'LSTM Gates', icon: '🚪' },
                    { id: 'gru', title: 'GRU', icon: '⚙️' },
                    { id: 'bidirectional', title: 'Bidirectional RNN', icon: '↔️' },
                ],
            },
            {
                id: 'transformers',
                title: 'Transformers & Attention',
                icon: '⚡',
                description:
                    'Self-attention lets the model weigh all input positions simultaneously. Transformers power GPT, BERT, and modern LLMs. The "Attention Is All You Need" paper revolutionized NLP and beyond.',
                children: [
                    { id: 'self-attention', title: 'Self-Attention', icon: '🔍' },
                    { id: 'multi-head', title: 'Multi-Head Attention', icon: '👥' },
                    { id: 'bert', title: 'BERT', icon: '📝' },
                    { id: 'gpt', title: 'GPT', icon: '💬' },
                    { id: 'vision-transformers', title: 'ViT', icon: '👁️' },
                ],
            },
            {
                id: 'gans',
                title: 'GANs (Generative Models)',
                icon: '🎨',
                description:
                    'Generator vs Discriminator in a min-max game. GANs generate realistic images, videos, and data. Variational Autoencoders (VAEs) and Diffusion Models are related generative approaches.',
                children: [
                    { id: 'generator', title: 'Generator Network', icon: '🎭' },
                    { id: 'discriminator', title: 'Discriminator', icon: '🔎' },
                    { id: 'vaes', title: 'VAEs', icon: '🔬' },
                    { id: 'diffusion', title: 'Diffusion Models', icon: '🌊' },
                ],
            },
        ],
    },
    {
        id: 'mlops',
        title: 'MLOps & Deployment',
        icon: '🚢',
        color: '#3b82f6',
        nodes: [
            {
                id: 'model-serving',
                title: 'Model Serving',
                icon: '🖥️',
                description:
                    'Deploy models as REST APIs with Flask/FastAPI, or use TensorFlow Serving / TorchServe. Understand latency vs throughput tradeoffs, batching, and model versioning.',
                children: [
                    { id: 'flask-fastapi', title: 'Flask / FastAPI', icon: '🌐' },
                    { id: 'tf-serving', title: 'TF Serving', icon: '📡' },
                    { id: 'torchserve', title: 'TorchServe', icon: '🔥' },
                    { id: 'onnx', title: 'ONNX Runtime', icon: '⚡' },
                ],
            },
            {
                id: 'monitoring',
                title: 'Model Monitoring',
                icon: '📡',
                description:
                    'Track data drift, concept drift, and model degradation in production. Set up alerts for prediction quality drops. Implement A/B testing and shadow deployments for safe rollouts.',
                children: [
                    { id: 'data-drift', title: 'Data Drift', icon: '📊' },
                    { id: 'concept-drift', title: 'Concept Drift', icon: '🔄' },
                    { id: 'ab-testing', title: 'A/B Testing', icon: '🧪' },
                ],
            },
            {
                id: 'cicd-ml',
                title: 'CI/CD for ML',
                icon: '🔄',
                description:
                    'Automate training, testing, and deployment pipelines with tools like MLflow, DVC, and Kubeflow. Version datasets and models alongside code. Reproducibility is non-negotiable in production ML.',
                children: [
                    { id: 'mlflow', title: 'MLflow', icon: '📦' },
                    { id: 'dvc', title: 'DVC', icon: '📂' },
                    { id: 'kubeflow', title: 'Kubeflow', icon: '🔧' },
                ],
            },
            {
                id: 'docker-k8s',
                title: 'Containerization & Scaling',
                icon: '🐳',
                description:
                    'Docker containers ensure reproducible environments. Kubernetes orchestrates scaling inference services. Understand resource requests, GPU scheduling, and horizontal pod autoscaling.',
                children: [
                    { id: 'docker', title: 'Docker', icon: '🐳' },
                    { id: 'kubernetes', title: 'Kubernetes', icon: '☸️' },
                    { id: 'gpu-scheduling', title: 'GPU Scheduling', icon: '🖥️' },
                ],
            },
        ],
    },
];

/* ═══════════════════════════════════════════════════════
   CURVED SVG CONNECTORS
   ═══════════════════════════════════════════════════════ */

const CurvedConnectors: React.FC<{
    isLeft: boolean;
    color: string;
    childCount: number;
}> = ({ isLeft, color, childCount }) => {
    const svgRef = useRef<SVGSVGElement>(null);
    const [curves, setCurves] = useState<
        { path: string; ex: number; ey: number; ox: number; oy: number }[]
    >([]);

    useEffect(() => {
        // Wait for the panel's scaleX animation (400ms) to finish before measuring
        const timer = setTimeout(() => {
            const svg = svgRef.current;
            if (!svg) return;
            const panel = svg.closest('.subbranch-panel') as HTMLElement;
            if (!panel) return;

            const cards = panel.querySelectorAll('.subbranch-card');
            if (cards.length === 0) return;

            const panelRect = panel.getBoundingClientRect();

            // Origin: center of the node card's outer edge
            // Panel top aligns with wrapper top (== node card top).
            // Node card is ~38px tall → center ≈ 19px
            const originY = 19;
            const originX = isLeft ? panelRect.width + 8 : -8;

            const newCurves: typeof curves = [];

            cards.forEach((card) => {
                const r = card.getBoundingClientRect();
                const cy = r.top + r.height / 2 - panelRect.top;
                const cx = isLeft
                    ? r.right - panelRect.left + 2
                    : r.left - panelRect.left - 2;

                // Smooth cubic bezier — control points create an organic S-curve
                const dx = Math.abs(cx - originX);
                const cpX = dx * 0.55;

                const d = isLeft
                    ? `M ${originX} ${originY} C ${originX - cpX} ${originY}, ${cx + cpX} ${cy}, ${cx} ${cy}`
                    : `M ${originX} ${originY} C ${originX + cpX} ${originY}, ${cx - cpX} ${cy}, ${cx} ${cy}`;

                newCurves.push({ path: d, ex: cx, ey: cy, ox: originX, oy: originY });
            });

            setCurves(newCurves);
        }, 450);

        return () => clearTimeout(timer);
    }, [childCount, isLeft]);

    return (
        <svg
            ref={svgRef}
            className="subbranch-svg"
            style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                overflow: 'visible',
                pointerEvents: 'none',
                zIndex: 0,
            }}
        >
            {curves.length > 0 && (
                <>
                    {/* ── Glow layer ── */}
                    {curves.map(({ path }, i) => (
                        <motion.path
                            key={`glow-${i}`}
                            d={path}
                            fill="none"
                            stroke={color}
                            strokeWidth={5}
                            strokeOpacity={0.07}
                            strokeLinecap="round"
                            initial={{ pathLength: 0 }}
                            animate={{ pathLength: 1 }}
                            transition={{
                                duration: 0.6,
                                delay: i * 0.07,
                                ease: [0.16, 1, 0.3, 1],
                            }}
                        />
                    ))}

                    {/* ── Main stroke ── */}
                    {curves.map(({ path }, i) => (
                        <motion.path
                            key={`main-${i}`}
                            d={path}
                            fill="none"
                            stroke={color}
                            strokeWidth={1.5}
                            strokeOpacity={0.35}
                            strokeLinecap="round"
                            initial={{ pathLength: 0 }}
                            animate={{ pathLength: 1 }}
                            transition={{
                                duration: 0.5,
                                delay: i * 0.07 + 0.05,
                                ease: [0.16, 1, 0.3, 1],
                            }}
                        />
                    ))}

                    {/* ── Origin dot (on the node card edge) ── */}
                    <motion.circle
                        cx={curves[0].ox}
                        cy={curves[0].oy}
                        r={3.5}
                        fill={color}
                        fillOpacity={0.5}
                        initial={{ scale: 0, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        transition={{ duration: 0.3 }}
                    />

                    {/* ── End dots (on each child card) ── */}
                    {curves.map(({ ex, ey }, i) => (
                        <motion.circle
                            key={`dot-${i}`}
                            cx={ex}
                            cy={ey}
                            r={2.5}
                            fill={color}
                            fillOpacity={0.45}
                            initial={{ scale: 0, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            transition={{
                                duration: 0.3,
                                delay: i * 0.07 + 0.3,
                            }}
                        />
                    ))}
                </>
            )}
        </svg>
    );
};

/* ═══════════════════════════════════════════════════════
   COMPONENT
   ═══════════════════════════════════════════════════════ */

const Roadmap: React.FC = () => {
    const [selectedNode, setSelectedNode] = useState<{
        node: RoadmapNode;
        category: RoadmapCategory;
    } | null>(null);

    // Only one expanded sub-branch at a time
    const [expandedNodeId, setExpandedNodeId] = useState<string | null>(null);

    const spineRef = useRef<HTMLDivElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const [spineHeight, setSpineHeight] = useState(0);

    // Close popover on Escape
    useEffect(() => {
        const handleEsc = (e: KeyboardEvent) => {
            if (e.key === 'Escape') {
                setSelectedNode(null);
                setExpandedNodeId(null);
            }
        };
        window.addEventListener('keydown', handleEsc);
        return () => window.removeEventListener('keydown', handleEsc);
    }, []);

    // Animate spine height on scroll
    useEffect(() => {
        const handleScroll = () => {
            if (!containerRef.current) return;
            const rect = containerRef.current.getBoundingClientRect();
            const containerTop = rect.top;
            const containerHeight = rect.height;
            const windowHeight = window.innerHeight;

            const scrolled = windowHeight - containerTop;
            const progress = Math.min(Math.max(scrolled / containerHeight, 0), 1);
            setSpineHeight(progress * 100);
        };

        window.addEventListener('scroll', handleScroll, { passive: true });
        handleScroll();
        return () => window.removeEventListener('scroll', handleScroll);
    }, []);

    const handleNodeClick = useCallback(
        (node: RoadmapNode, category: RoadmapCategory) => {
            setSelectedNode({ node, category });
        },
        []
    );

    const handleToggleExpand = useCallback(
        (e: React.MouseEvent, nodeId: string) => {
            e.stopPropagation(); // Don't trigger popover
            setExpandedNodeId((prev) => (prev === nodeId ? null : nodeId));
        },
        []
    );

    return (
        <div className="roadmap-container" ref={containerRef}>
            {/* ── Header ── */}
            <motion.div
                className="roadmap-header"
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
            >
                <span className="roadmap-badge">Interactive Roadmap</span>
                <h1>
                    Machine Learning{' '}
                    <span className="gradient-text">Learning Path</span>
                </h1>
                <p className="roadmap-subtitle">
                    Scroll to explore the complete ML journey — from fundamentals to
                    production deployment. Click any node to learn more.
                </p>
            </motion.div>

            {/* ── Tree ── */}
            <div className="roadmap-tree">
                {/* Spine — grows as you scroll */}
                <div className="roadmap-spine-track">
                    <div
                        className="roadmap-spine"
                        ref={spineRef}
                        style={{ height: `${spineHeight}%` }}
                    />
                </div>

                {roadmapData.map((category) => (
                    <div key={category.id} className="roadmap-category">
                        {/* ── Milestone node ── */}
                        <motion.div
                            className="roadmap-milestone"
                            initial={{ opacity: 0, scale: 0.7 }}
                            whileInView={{ opacity: 1, scale: 1 }}
                            viewport={{ once: true, amount: 0.5 }}
                            transition={{
                                duration: 0.5,
                                ease: [0.16, 1, 0.3, 1],
                                delay: 0.1,
                            }}
                        >
                            <div
                                className="milestone-dot"
                                style={{
                                    background: `linear-gradient(135deg, ${category.color}, ${category.color}99)`,
                                    boxShadow: `0 0 24px ${category.color}40`,
                                }}
                            />
                            <div className="milestone-content">
                                <span className="milestone-icon">{category.icon}</span>
                                <h2 style={{ color: category.color }}>
                                    {category.title}
                                </h2>
                                <span className="milestone-count">
                                    {category.nodes.length} topics
                                </span>
                            </div>
                        </motion.div>

                        {/* ── Leaf nodes ── */}
                        <div className="roadmap-nodes">
                            {category.nodes.map((node, nodeIdx) => {
                                const isLeft = nodeIdx % 2 === 0;
                                const hasChildren = node.children && node.children.length > 0;
                                const isExpanded = expandedNodeId === node.id;

                                return (
                                    <div
                                        key={node.id}
                                        className={`roadmap-node-wrapper ${isLeft ? 'wrapper-left' : 'wrapper-right'}`}
                                    >
                                        <motion.div
                                            className={`roadmap-node ${isLeft ? 'node-left' : 'node-right'}`}
                                            initial={{
                                                opacity: 0,
                                                x: isLeft ? -60 : 60,
                                            }}
                                            whileInView={{ opacity: 1, x: 0 }}
                                            viewport={{ once: true, amount: 0.3 }}
                                            transition={{
                                                duration: 0.6,
                                                ease: [0.16, 1, 0.3, 1],
                                                delay: nodeIdx * 0.08,
                                            }}
                                            onClick={() =>
                                                handleNodeClick(node, category)
                                            }
                                            style={
                                                {
                                                    '--node-color': category.color,
                                                } as React.CSSProperties
                                            }
                                        >
                                            {/* Connector line to spine */}
                                            <div className="node-connector" />

                                            {/* Node card */}
                                            <div className={`node-card ${isExpanded ? 'node-card-expanded' : ''}`}>
                                                <span className="node-icon">
                                                    {node.icon}
                                                </span>
                                                <span className="node-title">
                                                    {node.title}
                                                </span>
                                                {hasChildren && (
                                                    <button
                                                        className={`node-expand-btn ${isExpanded ? 'expanded' : ''}`}
                                                        onClick={(e) =>
                                                            handleToggleExpand(e, node.id)
                                                        }
                                                        aria-label={isExpanded ? 'Collapse' : 'Expand sub-topics'}
                                                        style={
                                                            {
                                                                '--node-color': category.color,
                                                            } as React.CSSProperties
                                                        }
                                                    >
                                                        <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                                                            <path
                                                                d={isLeft ? 'M8.5 3L4.5 7L8.5 11' : 'M5.5 3L9.5 7L5.5 11'}
                                                                stroke="currentColor"
                                                                strokeWidth="1.8"
                                                                strokeLinecap="round"
                                                                strokeLinejoin="round"
                                                            />
                                                        </svg>
                                                    </button>
                                                )}
                                                {!hasChildren && (
                                                    <span className="node-arrow">→</span>
                                                )}
                                            </div>
                                        </motion.div>

                                        {/* ── Horizontal sub-branches ── */}
                                        <AnimatePresence>
                                            {isExpanded && hasChildren && (
                                                <motion.div
                                                    className={`subbranch-panel ${isLeft ? 'panel-left' : 'panel-right'}`}
                                                    initial={{ opacity: 0, scaleX: 0 }}
                                                    animate={{ opacity: 1, scaleX: 1 }}
                                                    exit={{ opacity: 0, scaleX: 0 }}
                                                    transition={{
                                                        duration: 0.4,
                                                        ease: [0.16, 1, 0.3, 1],
                                                    }}
                                                    style={
                                                        {
                                                            '--node-color': category.color,
                                                            transformOrigin: isLeft ? 'right center' : 'left center',
                                                        } as React.CSSProperties
                                                    }
                                                >
                                                    {/* ── Curved SVG connectors ── */}
                                                    <CurvedConnectors
                                                        isLeft={isLeft}
                                                        color={category.color}
                                                        childCount={node.children!.length}
                                                    />

                                                    <div className="subbranch-nodes">
                                                        {node.children!.map((child, childIdx) => (
                                                            <motion.div
                                                                key={child.id}
                                                                className="subbranch-node"
                                                                initial={{
                                                                    opacity: 0,
                                                                    x: isLeft ? 20 : -20,
                                                                }}
                                                                animate={{ opacity: 1, x: 0 }}
                                                                exit={{ opacity: 0, x: isLeft ? 20 : -20 }}
                                                                transition={{
                                                                    duration: 0.35,
                                                                    ease: [0.16, 1, 0.3, 1],
                                                                    delay: childIdx * 0.06,
                                                                }}
                                                                style={
                                                                    {
                                                                        '--node-color': category.color,
                                                                    } as React.CSSProperties
                                                                }
                                                            >
                                                                <div className="subbranch-card">
                                                                    <span className="subbranch-icon">
                                                                        {child.icon}
                                                                    </span>
                                                                    <span className="subbranch-title">
                                                                        {child.title}
                                                                    </span>
                                                                </div>
                                                            </motion.div>
                                                        ))}
                                                    </div>
                                                </motion.div>
                                            )}
                                        </AnimatePresence>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                ))}

                {/* ── Finish flag ── */}
                <motion.div
                    className="roadmap-finish"
                    initial={{ opacity: 0, scale: 0.5 }}
                    whileInView={{ opacity: 1, scale: 1 }}
                    viewport={{ once: true, amount: 0.5 }}
                    transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
                >
                    <div className="finish-icon">🏆</div>
                    <h3>ML Engineer</h3>
                    <p>You've mapped the entire journey!</p>
                </motion.div>
            </div>

            {/* ── Popover ── */}
            <AnimatePresence>
                {selectedNode && (
                    <motion.div
                        className="roadmap-overlay"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        transition={{ duration: 0.2 }}
                        onClick={() => setSelectedNode(null)}
                    >
                        <motion.div
                            className="roadmap-popover"
                            initial={{ opacity: 0, scale: 0.9, y: 20 }}
                            animate={{ opacity: 1, scale: 1, y: 0 }}
                            exit={{ opacity: 0, scale: 0.9, y: 20 }}
                            transition={{
                                duration: 0.35,
                                ease: [0.16, 1, 0.3, 1],
                            }}
                            onClick={(e) => e.stopPropagation()}
                        >
                            <div className="popover-header">
                                <div className="popover-meta">
                                    <span
                                        className="popover-category-badge"
                                        style={{
                                            color: selectedNode.category.color,
                                            background: `${selectedNode.category.color}15`,
                                            borderColor: `${selectedNode.category.color}30`,
                                        }}
                                    >
                                        {selectedNode.category.icon}{' '}
                                        {selectedNode.category.title}
                                    </span>
                                </div>
                                <button
                                    className="popover-close"
                                    onClick={() => setSelectedNode(null)}
                                    aria-label="Close"
                                >
                                    ✕
                                </button>
                            </div>
                            <div className="popover-body">
                                <div className="popover-icon-large">
                                    {selectedNode.node.icon}
                                </div>
                                <h3>{selectedNode.node.title}</h3>
                                <p>{selectedNode.node.description}</p>
                            </div>
                        </motion.div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

export default Roadmap;
