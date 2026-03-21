import React from 'react';
import { useAuth } from '../context/AuthContext';
import { useNavigate, useLocation } from 'react-router-dom';
import {
    ShieldCheck, LogOut, Upload, FileText, BarChart3, Brain, Home,
} from 'lucide-react';

const NAV_ITEMS = [
    { path: '/', label: 'Dashboard', icon: Home },
    { path: '/upload', label: 'Upload', icon: Upload },
    { path: '/documents', label: 'Documents', icon: FileText },
    { path: '/analysis', label: 'Analysis', icon: BarChart3 },
    { path: '/chat', label: 'AI Chat', icon: Brain },
];

const Navbar = () => {
    const { user, logout } = useAuth();
    const navigate = useNavigate();
    const location = useLocation();

    return (
        <div className="pt-4 px-4 sticky top-0 z-50">
            <header className="max-w-7xl mx-auto bg-white/70 backdrop-blur-md border border-white/20 rounded-2xl shadow-[0_8px_30px_rgb(0,0,0,0.04)] transition-all">
                <div className="h-16 flex items-center justify-between px-6">
                    {/* Logo */}
                    <div
                        className="flex items-center gap-3 cursor-pointer group"
                        onClick={() => navigate('/')}
                    >
                        <div className="bg-gradient-to-br from-indigo-500 to-indigo-700 p-2 rounded-xl shadow-md shadow-indigo-500/20 group-hover:shadow-indigo-500/40 transition-shadow">
                            <ShieldCheck className="w-5 h-5 text-white" />
                        </div>
                        <h1 className="text-lg font-display font-bold text-slate-900 hidden sm:block tracking-tight">
                            Intelligent Policy Analyzer
                        </h1>
                    </div>

                    {/* Nav Links */}
                    <nav className="flex items-center gap-1.5 bg-slate-100/50 p-1 rounded-xl">
                        {NAV_ITEMS.map(item => {
                            const isActive = location.pathname === item.path;
                            return (
                                <button
                                    key={item.path}
                                    onClick={() => navigate(item.path)}
                                    className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                                        isActive
                                            ? 'bg-white text-indigo-700 shadow-sm border border-slate-200/60'
                                            : 'text-slate-500 hover:bg-slate-200/50 hover:text-slate-800'
                                    }`}
                                >
                                    <item.icon className={`w-4 h-4 ${isActive ? 'text-indigo-600' : ''}`} />
                                    <span className="hidden md:inline">{item.label}</span>
                                </button>
                            );
                        })}
                    </nav>

                    {/* User */}
                    <div className="flex items-center gap-4">
                        <div className="hidden sm:flex flex-col items-end">
                            <span className="text-sm font-semibold text-slate-800 leading-tight">
                                {user?.full_name || user?.username}
                            </span>
                            <span className="text-[10px] text-slate-400 font-medium uppercase tracking-wider">
                                {user?.email ? 'Member' : 'Admin'}
                            </span>
                        </div>
                        <div className="h-8 w-px bg-slate-200 hidden sm:block"></div>
                        <button
                            onClick={logout}
                            className="text-slate-400 hover:text-red-600 transition-colors p-2 rounded-xl hover:bg-red-50"
                            title="Sign out"
                        >
                            <LogOut className="w-5 h-5" />
                        </button>
                    </div>
                </div>
            </header>
        </div>
    );
};

export default Navbar;
