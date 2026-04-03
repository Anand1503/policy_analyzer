import React from 'react';
import { useAuth } from '../context/AuthContext';
import { useTheme } from '../context/ThemeContext';
import { useNavigate, useLocation } from 'react-router-dom';
import {
    ShieldCheck, LogOut, LayoutDashboard,
    Upload, Clock, MessageSquare, Moon, Sun
} from 'lucide-react';

const NAV_ITEMS = [
    { path: '/upload', label: 'Upload', icon: Upload },
    { path: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { path: '/history', label: 'History', icon: Clock },
    { path: '/chat', label: 'AI Chat', icon: MessageSquare },
];

const Navbar = () => {
    const { user, logout } = useAuth();
    const { theme, toggleTheme } = useTheme();
    const navigate = useNavigate();
    const location = useLocation();

    const isDark = theme === 'dark';

    const sidebarBg = isDark ? '#1b263b' : '#0d1b2a';
    const activeBg = isDark ? 'rgba(119,141,169,0.2)' : 'rgba(255,255,255,0.12)';
    const hoverBg = isDark ? 'rgba(119,141,169,0.1)' : 'rgba(255,255,255,0.07)';
    const textColor = '#e0e1dd';
    const mutedColor = '#778da9';

    return (
        <aside
            style={{
                width: '240px',
                minHeight: '100vh',
                background: sidebarBg,
                display: 'flex',
                flexDirection: 'column',
                padding: '0',
                position: 'fixed',
                top: 0,
                left: 0,
                bottom: 0,
                zIndex: 50,
                borderRight: '1px solid rgba(119,141,169,0.15)',
            }}
        >
            {/* Logo */}
            <div style={{
                padding: '28px 24px 20px',
                borderBottom: '1px solid rgba(119,141,169,0.1)',
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
            }}>
                <div style={{
                    width: '40px', height: '40px', borderRadius: '12px',
                    background: 'linear-gradient(135deg, #415a77, #778da9)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    flexShrink: 0,
                }}>
                    <ShieldCheck size={22} color="white" />
                </div>
                <div>
                    <div style={{ fontSize: '18px', fontWeight: '700', color: textColor, lineHeight: 1 }}>PolicyAI</div>
                    <div style={{ fontSize: '11px', color: mutedColor, marginTop: '2px' }}>Policy Analyzer</div>
                </div>
            </div>

            {/* User Info */}
            <div style={{
                padding: '16px 20px',
                borderBottom: '1px solid rgba(119,141,169,0.1)',
            }}>
                <div style={{
                    background: 'rgba(119,141,169,0.1)',
                    borderRadius: '10px',
                    padding: '12px',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '10px',
                }}>
                    <div style={{
                        width: '34px', height: '34px', borderRadius: '50%',
                        background: 'linear-gradient(135deg, #415a77, #778da9)',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        fontSize: '15px', fontWeight: '700', color: 'white',
                        flexShrink: 0,
                    }}>
                        {(user?.full_name || user?.username || 'U')[0].toUpperCase()}
                    </div>
                    <div style={{ overflow: 'hidden' }}>
                        <div style={{
                            fontSize: '14px', fontWeight: '600', color: textColor,
                            whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
                        }}>
                            {user?.full_name || user?.username}
                        </div>
                        <div style={{ fontSize: '11px', color: mutedColor }}>Logged in</div>
                    </div>
                </div>
            </div>

            {/* Navigation Items */}
            <nav style={{ flex: 1, padding: '16px 12px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                <div style={{ fontSize: '11px', fontWeight: '600', color: mutedColor, textTransform: 'uppercase', letterSpacing: '0.08em', padding: '0 8px', marginBottom: '8px' }}>
                    Navigation
                </div>
                {NAV_ITEMS.map(item => {
                    const isActive = location.pathname === item.path;
                    return (
                        <button
                            key={item.path}
                            onClick={() => navigate(item.path)}
                            style={{
                                width: '100%',
                                display: 'flex',
                                alignItems: 'center',
                                gap: '12px',
                                padding: '12px 14px',
                                borderRadius: '10px',
                                border: 'none',
                                cursor: 'pointer',
                                background: isActive ? activeBg : 'transparent',
                                color: isActive ? textColor : mutedColor,
                                fontSize: '15px',
                                fontWeight: isActive ? '600' : '500',
                                textAlign: 'left',
                                transition: 'all 0.15s',
                                borderLeft: isActive ? '3px solid #778da9' : '3px solid transparent',
                            }}
                            onMouseEnter={e => { if (!isActive) e.currentTarget.style.background = hoverBg; }}
                            onMouseLeave={e => { if (!isActive) e.currentTarget.style.background = 'transparent'; }}
                        >
                            <item.icon size={20} />
                            <span>{item.label}</span>
                        </button>
                    );
                })}
            </nav>

            {/* Dark Mode Toggle */}
            <div style={{
                padding: '16px 12px',
                borderTop: '1px solid rgba(119,141,169,0.1)',
            }}>
                <div style={{
                    padding: '8px 14px',
                    borderRadius: '10px',
                    background: 'rgba(119,141,169,0.08)',
                    marginBottom: '8px',
                }}>
                    <div style={{ fontSize: '11px', color: mutedColor, marginBottom: '10px', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                        Appearance
                    </div>
                    {/* Toggle Switch */}
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            {isDark ? <Moon size={16} color="#778da9" /> : <Sun size={16} color="#f59e0b" />}
                            <span style={{ fontSize: '14px', color: textColor, fontWeight: '500' }}>
                                {isDark ? 'Dark Mode' : 'Light Mode'}
                            </span>
                        </div>
                        {/* Toggle button */}
                        <button
                            onClick={toggleTheme}
                            title="Toggle theme"
                            style={{
                                width: '44px', height: '24px',
                                borderRadius: '12px',
                                border: 'none',
                                cursor: 'pointer',
                                background: isDark ? '#415a77' : 'rgba(119,141,169,0.3)',
                                position: 'relative',
                                transition: 'background 0.25s',
                                padding: 0,
                                flexShrink: 0,
                            }}
                        >
                            <div style={{
                                position: 'absolute',
                                top: '3px',
                                left: isDark ? '23px' : '3px',
                                width: '18px', height: '18px',
                                borderRadius: '50%',
                                background: 'white',
                                transition: 'left 0.25s',
                                boxShadow: '0 1px 4px rgba(0,0,0,0.3)',
                            }} />
                        </button>
                    </div>
                </div>

                {/* Logout */}
                <button
                    onClick={logout}
                    style={{
                        width: '100%',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '12px',
                        padding: '12px 14px',
                        borderRadius: '10px',
                        border: 'none',
                        cursor: 'pointer',
                        background: 'transparent',
                        color: '#ef4444',
                        fontSize: '15px',
                        fontWeight: '500',
                        textAlign: 'left',
                        transition: 'all 0.15s',
                    }}
                    onMouseEnter={e => { e.currentTarget.style.background = 'rgba(239,68,68,0.1)'; }}
                    onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; }}
                >
                    <LogOut size={20} />
                    <span>Sign Out</span>
                </button>
            </div>
        </aside>
    );
};

export default Navbar;
