import React, { useEffect } from 'react';
import { NavLink, Outlet, useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import Footer from './Footer';

const navLinks = [
    { to: '/dataset', label: 'Dataset' },
    { to: '/architect', label: 'Architect' },
    { to: '/training', label: 'Training' },
    { to: '/learn', label: 'Learn' },
];

const Layout: React.FC = () => {
    const { user, setUser } = useAuth();
    const navigate = useNavigate();
    const location = useLocation();

    const isHome = location.pathname === '/';

    // Scroll to top on route change
    useEffect(() => {
        window.scrollTo(0, 0);
    }, [location.pathname]);

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
        <div className="flex flex-col min-h-screen">
            {/* ── Liquid Glass Navbar ── */}
            <nav
                className={[
                    'fixed top-5 left-1/2 -translate-x-1/2 z-50',
                    'flex items-center justify-between',
                    'px-5 py-2.5 rounded-2xl',
                    'w-[92%] max-w-4xl',
                    // Liquid glass surface
                    'bg-white/[0.06]',
                    'backdrop-blur-2xl',
                    'border border-white/[0.12]',
                    // Inner glass highlight (top edge glow)
                    'shadow-[inset_0_1px_1px_rgba(255,255,255,0.1),inset_0_-1px_1px_rgba(255,255,255,0.04)]',
                    // Outer ambient glow — soft purple-blue to make glass visible
                    'shadow-[0_8px_40px_-8px_rgba(120,100,255,0.12),0_2px_12px_rgba(0,0,0,0.4)]',
                    'transition-all duration-500 ease-out',
                ].join(' ')}
                style={{
                    // CSS lets us layer multiple box-shadows that Tailwind merges weirdly
                    boxShadow: [
                        'inset 0 1px 1px rgba(255,255,255,0.1)',
                        'inset 0 -1px 1px rgba(255,255,255,0.04)',
                        '0 8px 40px -8px rgba(120,100,255,0.15)',
                        '0 2px 12px rgba(0,0,0,0.45)',
                    ].join(', '),
                }}
            >
                {/* ── Logo ── */}
                <NavLink to="/" className="flex items-center gap-2 shrink-0 group no-underline">
                    <span className="text-lg select-none transition-transform duration-300 group-hover:rotate-90">◎</span>
                    <span className="text-[15px] font-semibold text-white tracking-tight">
                        Vortex<span className="text-accent-2">ML</span>
                    </span>
                </NavLink>

                {/* ── Center Nav Links ── */}
                <div className="flex items-center gap-0.5 mx-auto">
                    {navLinks.map(({ to, label }) => (
                        <NavLink
                            key={to}
                            to={to}
                            className={({ isActive }) =>
                                [
                                    'px-4 py-1.5 rounded-xl text-[13px] font-medium no-underline',
                                    'transition-all duration-300 ease-out',
                                    isActive
                                        ? 'bg-white/[0.12] text-white shadow-[inset_0_1px_0_rgba(255,255,255,0.08)]'
                                        : 'text-white/55 hover:text-white/90 hover:bg-white/[0.06]',
                                ].join(' ')
                            }
                        >
                            {label}
                        </NavLink>
                    ))}
                </div>

                {/* ── Auth Section ── */}
                <div className="flex items-center shrink-0">
                    {user ? (
                        <div className="flex items-center gap-3">
                            <span className="hidden md:inline text-[13px] text-white/60 font-medium">
                                {user.username}
                                <span className="ml-2 px-2 py-0.5 rounded-lg bg-white/[0.07] text-[11px] text-white/45 border border-white/[0.06]">
                                    {user.is_beginner ? '🌿 Novice' : '🔥 Expert'}
                                </span>
                            </span>
                            <button
                                onClick={handleLogout}
                                className="px-3.5 py-1.5 text-[13px] font-medium text-white/90 hover:text-white rounded-xl border border-white/10 bg-white/[0.05] hover:bg-white/[0.1] transition-all duration-300 focus:outline-none"
                            >
                                Logout
                            </button>
                        </div>
                    ) : (
                        <div className="flex items-center gap-1.5">
                            <NavLink
                                to="/signin"
                                className="px-4 py-1.5 text-[13px] font-medium text-white/50 hover:text-white/90 rounded-xl hover:bg-white/[0.06] transition-all duration-300 focus:outline-none no-underline"
                            >
                                Sign In
                            </NavLink>
                            <NavLink
                                to="/signup"
                                className="px-4 py-1.5 text-[13px] font-semibold text-white rounded-xl bg-gradient-to-r from-accent-1 to-accent-2 hover:opacity-90 hover:scale-[1.03] active:scale-[0.98] transition-all duration-300 focus:outline-none shadow-[0_0_16px_rgba(139,92,246,0.3)] no-underline"
                            >
                                Sign Up
                            </NavLink>
                        </div>
                    )}
                </div>
            </nav>

            <main className={isHome ? 'flex-grow relative' : 'main-content flex-grow'}>
                <Outlet />
            </main>
            <Footer />
        </div>
    );
};

export default Layout;
