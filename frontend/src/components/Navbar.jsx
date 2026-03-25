import React from 'react';
import { useAuth } from '../context/AuthContext';
import { useNavigate, useLocation } from 'react-router-dom';
import { ShieldCheck, LogOut, LayoutDashboard, Upload, Clock, MessageSquare } from 'lucide-react';

const NAV_ITEMS = [
    { path: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { path: '/upload', label: 'Upload', icon: Upload },
    { path: '/history', label: 'History', icon: Clock },
    { path: '/chat', label: 'Chat', icon: MessageSquare },
];

const Navbar = () => {
    const { user, logout } = useAuth();
    const navigate = useNavigate();
    const location = useLocation();

    return (
        <div className="pt-3 px-4 sticky top-0 z-50">
            <header className="max-w-6xl mx-auto rounded-2xl border"
                style={{
                    background: 'rgba(224,225,221,0.85)',
                    backdropFilter: 'blur(16px)',
                    borderColor: 'rgba(119,141,169,0.15)',
                    boxShadow: '0 2px 20px rgba(13,27,42,0.06)',
                }}>
                <div className="h-14 flex items-center justify-between px-5">
                    {/* Logo */}
                    <div className="flex items-center gap-2.5 cursor-pointer" onClick={() => navigate('/')}>
                        <div className="w-8 h-8 rounded-lg flex items-center justify-center"
                            style={{ background: 'linear-gradient(135deg, #415a77, #778da9)' }}>
                            <ShieldCheck className="w-4 h-4 text-white" />
                        </div>
                        <h1 className="text-[15px] font-display font-bold hidden sm:block" style={{ color: '#0d1b2a' }}>
                            PolicyAI
                        </h1>
                    </div>

                    {/* Nav Links */}
                    <nav className="flex items-center gap-1 p-1 rounded-xl" style={{ background: 'rgba(13,27,42,0.04)' }}>
                        {NAV_ITEMS.map(item => {
                            const isActive = location.pathname === item.path;
                            return (
                                <button
                                    key={item.path}
                                    onClick={() => navigate(item.path)}
                                    className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-[13px] font-medium transition-all"
                                    style={{
                                        background: isActive ? 'white' : 'transparent',
                                        color: isActive ? '#0d1b2a' : '#778da9',
                                        boxShadow: isActive ? '0 1px 3px rgba(13,27,42,0.08)' : 'none',
                                    }}
                                >
                                    <item.icon className="w-3.5 h-3.5" style={{ color: isActive ? '#415a77' : '#778da9' }} />
                                    <span className="hidden md:inline">{item.label}</span>
                                </button>
                            );
                        })}
                    </nav>

                    {/* User & Logout */}
                    <div className="flex items-center gap-3">
                        <span className="text-[13px] font-medium hidden sm:block" style={{ color: '#415a77' }}>
                            {user?.username}
                        </span>
                        <button
                            onClick={logout}
                            className="p-1.5 rounded-lg transition-colors"
                            style={{ color: '#778da9' }}
                            onMouseEnter={e => { e.currentTarget.style.color = '#b91c1c'; e.currentTarget.style.background = '#fef2f2'; }}
                            onMouseLeave={e => { e.currentTarget.style.color = '#778da9'; e.currentTarget.style.background = 'transparent'; }}
                            title="Sign out"
                        >
                            <LogOut className="w-4 h-4" />
                        </button>
                    </div>
                </div>
            </header>
        </div>
    );
};

export default Navbar;
