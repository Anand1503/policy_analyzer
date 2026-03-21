import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { ShieldCheck, LogIn, UserPlus, ArrowRight, Brain, FileText, Lock } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const LoginPage = () => {
    const { login, register } = useAuth();
    const [isRegister, setIsRegister] = useState(false);
    const [form, setForm] = useState({ username: '', password: '', email: '', full_name: '' });
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setLoading(true);
        try {
            if (isRegister) {
                await register(form);
            } else {
                await login(form.username, form.password);
            }
        } catch (err) {
            setError(err.response?.data?.detail || 'Authentication failed');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen flex selection:bg-indigo-500/30">
            {/* LEFT SIDE - Animated Gradient / Branding Visuals */}
            <div className="hidden lg:flex lg:w-1/2 relative overflow-hidden bg-slate-900 border-r border-white/10">
                {/* Dynamic animated glow background */}
                <div className="absolute top-[-20%] left-[-10%] w-[140%] h-[140%] bg-[radial-gradient(circle,rgba(79,70,229,0.15)_0%,rgba(15,23,42,1)_60%)] animate-[pulse_10s_ease-in-out_infinite]" />
                <div className="absolute bottom-[-20%] right-[-10%] w-[120%] h-[120%] bg-[radial-gradient(circle,rgba(59,130,246,0.15)_0%,rgba(15,23,42,0)_60%)] animate-[pulse_12s_ease-in-out_infinite_reverse]" />
                
                {/* Grid Pattern overlay */}
                <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAiIGhlaWdodD0iNDAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik0wIDBINDBWMHoiIGZpbGw9Im5vbmUiLz4KPHBhdGggZD0iTTAgNDBoNDBNNDAgMHY0MCIgc3Ryb2tlPSJyZ2JhKDI1NSwyNTUsMjU1LDAuMDUpIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiLz4KPC9zdmc+')] opacity-50" />

                <div className="relative z-10 flex flex-col justify-between h-full p-16">
                    <motion.div 
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.8 }}
                        className="flex items-center gap-3"
                    >
                        <div className="flex items-center justify-center w-12 h-12 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl shadow-[0_0_20px_rgba(79,70,229,0.4)]">
                            <ShieldCheck className="w-6 h-6 text-white" />
                        </div>
                        <span className="text-xl font-display font-bold text-white tracking-wide">
                            Intelligent Policy Analyzer
                        </span>
                    </motion.div>

                    <div className="max-w-md">
                        <motion.h1 
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.8, delay: 0.2 }}
                            className="text-[3.25rem] font-display font-extrabold text-white leading-[1.05] tracking-tight mb-6"
                        >
                            AI-Powered <br/>
                            <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-indigo-400 drop-shadow-sm">
                                Privacy Intelligence
                            </span>
                        </motion.h1>
                        
                        <motion.p 
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.8, delay: 0.3 }}
                            className="text-[1.1rem] text-slate-300/90 mb-10 leading-relaxed font-normal tracking-wide"
                        >
                            Instantly analyze legal documents, detect hidden risks, and map GDPR/CCPA compliance using state-of-the-art NLP models.
                        </motion.p>
                        
                        <motion.div 
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            transition={{ duration: 1, delay: 0.6 }}
                            className="flex flex-col gap-4"
                        >
                            {[
                                { icon: Brain, text: "Context-aware Clause Extraction" },
                                { icon: Lock, text: "Regulatory Compliance Mapping" },
                                { icon: FileText, text: "Plain-English Summarization" }
                            ].map((feature, i) => (
                                <div key={i} className="flex items-center gap-3 text-slate-300">
                                    <div className="flex-shrink-0 w-8 h-8 rounded-full bg-white/5 flex items-center justify-center border border-white/10">
                                        <feature.icon className="w-4 h-4 text-indigo-400" />
                                    </div>
                                    <span className="font-medium text-sm">{feature.text}</span>
                                </div>
                            ))}
                        </motion.div>
                    </div>

                    <div className="text-xs text-slate-500 font-medium">
                        © 2026 Policy Optimizer AI. All rights reserved.
                    </div>
                </div>
            </div>

            {/* RIGHT SIDE - Form Container */}
            <div className="w-full lg:w-1/2 flex items-center justify-center bg-white relative">
                {/* Mobile branding (visible only on small screens) */}
                <div className="absolute top-8 left-8 flex lg:hidden items-center gap-2">
                    <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center">
                        <ShieldCheck className="w-4 h-4 text-white" />
                    </div>
                    <span className="font-display font-bold text-slate-900 tracking-tight">IPA</span>
                </div>

                <div className="w-full max-w-md px-8 py-12">
                    <AnimatePresence mode="wait">
                        <motion.div
                            key={isRegister ? 'register' : 'login'}
                            initial={{ opacity: 0, x: 10 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: -10 }}
                            transition={{ duration: 0.3 }}
                        >
                            <div className="text-center mb-10">
                                <h2 className="text-[2rem] font-display font-bold text-slate-900 mb-2 tracking-tight">
                                    {isRegister ? 'Create an account' : 'Welcome back'}
                                </h2>
                                <p className="text-slate-500 font-medium tracking-wide">
                                    {isRegister 
                                        ? 'Start analyzing your policies today.' 
                                        : 'Please enter your credentials to continue.'}
                                </p>
                            </div>

                            <form onSubmit={handleSubmit} className="space-y-5">
                                {isRegister && (
                                    <motion.div 
                                        initial={{ opacity: 0, height: 0 }}
                                        animate={{ opacity: 1, height: 'auto' }}
                                        className="space-y-5"
                                    >
                                        <div>
                                            <label className="block text-sm font-medium text-slate-700 mb-1.5 ml-1">Full Name</label>
                                            <input
                                                type="text"
                                                placeholder="John Doe"
                                                value={form.full_name}
                                                onChange={(e) => setForm({ ...form, full_name: e.target.value })}
                                                className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-slate-900 placeholder-slate-400 focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 focus:bg-white outline-none transition-all shadow-sm"
                                            />
                                        </div>
                                        <div>
                                            <label className="block text-sm font-medium text-slate-700 mb-1.5 ml-1">Email <span className="text-red-500">*</span></label>
                                            <input
                                                type="email"
                                                placeholder="john@example.com"
                                                value={form.email}
                                                onChange={(e) => setForm({ ...form, email: e.target.value })}
                                                className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-slate-900 placeholder-slate-400 focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 focus:bg-white outline-none transition-all shadow-sm"
                                                required
                                            />
                                        </div>
                                    </motion.div>
                                )}
                                
                                <div>
                                    <label className="block text-sm font-medium text-slate-700 mb-1.5 ml-1">Username <span className="text-red-500">*</span></label>
                                    <input
                                        type="text"
                                        placeholder="Enter your username"
                                        value={form.username}
                                        onChange={(e) => setForm({ ...form, username: e.target.value })}
                                        className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-slate-900 placeholder-slate-400 focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 focus:bg-white outline-none transition-all shadow-sm"
                                        required
                                    />
                                </div>
                                
                                <div>
                                    <div className="flex justify-between items-center mb-1.5 ml-1">
                                        <label className="block text-sm font-medium text-slate-700">Password <span className="text-red-500">*</span></label>
                                        {!isRegister && (
                                            <a href="#" className="text-xs font-medium text-indigo-600 hover:text-indigo-700 transition-colors">
                                                Forgot password?
                                            </a>
                                        )}
                                    </div>
                                    <input
                                        type="password"
                                        placeholder="••••••••"
                                        value={form.password}
                                        onChange={(e) => setForm({ ...form, password: e.target.value })}
                                        className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-slate-900 placeholder-slate-400 focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 focus:bg-white outline-none transition-all shadow-sm"
                                        required
                                    />
                                </div>

                                {error && (
                                    <motion.div 
                                        initial={{ opacity: 0, y: -5 }} animate={{ opacity: 1, y: 0 }}
                                        className="bg-red-50 text-red-600 text-sm p-4 rounded-xl border border-red-100 flex items-start gap-2"
                                    >
                                        <div className="mt-0.5">⚠️</div>
                                        <div>{error}</div>
                                    </motion.div>
                                )}

                                <button
                                    type="submit"
                                    disabled={loading}
                                    className="relative w-full bg-slate-900 hover:bg-indigo-600 text-white py-3.5 rounded-xl font-medium transition-all duration-300 shadow-[0_4px_14px_0_rgba(15,23,42,0.39)] hover:shadow-[0_6px_20px_rgba(79,70,229,0.23)] hover:-translate-y-0.5 disabled:opacity-70 disabled:hover:translate-y-0 disabled:hover:shadow-none overflow-hidden group mt-4 flex items-center justify-center gap-2"
                                >
                                    {loading ? (
                                        <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></div>
                                    ) : (
                                        <>
                                            {isRegister ? 'Create Account' : 'Sign In'}
                                            <ArrowRight className="w-4 h-4 opacity-70 group-hover:opacity-100 group-hover:translate-x-1 transition-all" />
                                        </>
                                    )}
                                </button>
                            </form>

                            <div className="mt-8 text-center pt-6 border-t border-slate-100">
                                <p className="text-slate-500 text-sm">
                                    {isRegister ? 'Already have an account? ' : "Don't have an account? "}
                                    <button
                                        onClick={() => { setIsRegister(!isRegister); setError(''); setForm({ username: '', password: '', email: '', full_name: '' }); }}
                                        className="font-semibold text-indigo-600 hover:text-indigo-700 transition-colors focus:outline-none"
                                    >
                                        {isRegister ? 'Sign In' : "Sign Up"}
                                    </button>
                                </p>
                            </div>
                        </motion.div>
                    </AnimatePresence>
                </div>
            </div>
        </div>
    );
};

export default LoginPage;
