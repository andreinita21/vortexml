import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const Signin: React.FC = () => {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState<string | null>(null);
    const { checkAuth } = useAuth();
    const navigate = useNavigate();

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError(null);

        try {
            const res = await fetch('/api/auth/signin', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password })
            });

            const data = await res.json();
            if (res.ok) {
                await checkAuth(); // Refetch user globally
                navigate('/');
            } else {
                setError(data.error || 'Authentication failed');
            }
        } catch (err) {
            setError('Network error. Please try again.');
        }
    };

    return (
        <div className="auth-page page-container fade-in">
            <div className="auth-card glass-panel interactive-card">
                <div className="card-header">
                    <div className="icon-wrapper">
                        <span className="card-icon">⚡</span>
                    </div>
                    <h2>Welcome Back</h2>
                    <p>Enter your credentials to access Vortex ML.</p>
                </div>

                <form className="auth-form" onSubmit={handleSubmit}>
                    {error && <div className="error-message">{error}</div>}

                    <div className="form-group">
                        <label>Email Address</label>
                        <input
                            type="email"
                            className="glass-input"
                            placeholder="neural@vortex.ai"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            required
                        />
                    </div>

                    <div className="form-group">
                        <label>Password</label>
                        <input
                            type="password"
                            className="glass-input"
                            placeholder="••••••••"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            required
                        />
                    </div>

                    <button type="submit" className="btn-primary w-full pulse-glow">Sign In</button>

                    <div className="auth-footer">
                        <span>Don't have an account?</span>
                        <Link to="/signup" className="auth-link">Sign Up</Link>
                    </div>
                </form>
            </div>
        </div>
    );
};

export default Signin;
