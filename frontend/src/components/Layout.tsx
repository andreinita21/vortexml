import React from 'react';
import { NavLink, Outlet, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const Layout: React.FC = () => {
    const { user, setUser } = useAuth();
    const navigate = useNavigate();

    const handleLogout = async () => {
        try {
            await fetch('/api/auth/logout', { method: 'POST' });
            setUser(null);
            navigate('/');
        } catch (error) {
            console.error("Failed to logout:", error);
        }
    };

    return (
        <>
            <nav className="navbar">
                <NavLink to="/" className="nav-logo">
                    <span className="logo-icon">◎</span>
                    <span className="logo-text">Vortex<span className="logo-accent">ML</span></span>
                </NavLink>
                <div className="nav-links">
                    <NavLink to="/courses" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
                        <span className="nav-icon">🎓</span> Courses
                    </NavLink>
                    <NavLink to="/dataset" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
                        <span className="nav-icon">📊</span> Dataset
                    </NavLink>
                    <NavLink to="/architect" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
                        <span className="nav-icon">🧠</span> Architect
                    </NavLink>
                    <NavLink to="/training" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
                        <span className="nav-icon">⚡</span> Training
                    </NavLink>
                </div>

                <div className="nav-auth">
                    {user ? (
                        <div className="user-menu">
                            <span className="welcome-text">Hi, {user.username} <span className="rank-badge">{user.is_beginner ? "🌿 Novice" : "🔥 Expert"}</span></span>
                            <button onClick={handleLogout} className="btn-secondary btn-sm auth-btn">Logout</button>
                        </div>
                    ) : (
                        <div className="auth-links">
                            <NavLink to="/signin" className="btn-secondary btn-sm auth-btn">Sign In</NavLink>
                            <NavLink to="/signup" className="btn-primary btn-sm auth-btn pulse-glow">Sign Up</NavLink>
                        </div>
                    )}
                </div>
            </nav>
            <main className="main-content">
                <Outlet />
            </main>
        </>
    );
};

export default Layout;
