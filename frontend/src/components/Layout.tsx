import React from 'react';
import { NavLink, Outlet } from 'react-router-dom';

const Layout: React.FC = () => {
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
            </nav>
            <main className="main-content">
                <Outlet />
            </main>
        </>
    );
};

export default Layout;
