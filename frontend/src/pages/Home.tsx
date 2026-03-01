import React from 'react';
import { Link } from 'react-router-dom';

const Home: React.FC = () => {
    return (
        <>
            <section className="hero">
                <div className="hero-badge">✦ Open Source ML Platform</div>
                <h1>Build Neural Networks<br /><span className="gradient-text">Without the Complexity</span></h1>
                <p className="hero-subtitle">
                    Upload your data, pick an architecture, and watch your model train in real-time.
                    No boilerplate. No PhD required.
                </p>
                <Link to="/dataset" className="hero-cta">
                    Get Started <span>→</span>
                </Link>
            </section>

            <section className="features-grid">
                <Link to="/dataset" className="feature-card" style={{ animationDelay: '0.1s' }}>
                    <div className="card-icon">📊</div>
                    <h3>Dataset Designer</h3>
                    <p>Upload CSV or Excel files. Preview your data, select input features and prediction targets with a visual column picker.</p>
                    <div className="card-arrow">Open Designer <span>→</span></div>
                </Link>
                <Link to="/architect" className="feature-card" style={{ animationDelay: '0.2s' }}>
                    <div className="card-icon">🧠</div>
                    <h3>Architecture Builder</h3>
                    <p>Choose from 10 neural network architectures. Configure layers, neurons, and hyperparameters with an intuitive interface.</p>
                    <div className="card-arrow">Build Architecture <span>→</span></div>
                </Link>
                <Link to="/training" className="feature-card" style={{ animationDelay: '0.3s' }}>
                    <div className="card-icon">⚡</div>
                    <h3>Live Training</h3>
                    <p>Watch your model learn in real-time with live loss charts, network visualization, and estimated time remaining.</p>
                    <div className="card-arrow">View Dashboard <span>→</span></div>
                </Link>
            </section>
        </>
    );
};

export default Home;
