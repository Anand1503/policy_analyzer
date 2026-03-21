import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { ShieldCheck, ArrowRight, Brain, FileText, Lock } from 'lucide-react';
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

    const inputClass = "w-full border rounded-xl px-4 py-3 text-sm outline-none transition-all font-sans placeholder:font-sans";

    return (
        <div className="min-h-screen flex" style={{ background: '#0d1b2a' }}>
            {/* ── Left Panel: Branding ──────────────────── */}
            <div className="hidden lg:flex lg:w-1/2 relative overflow-hidden" style={{ background: '#1b263b' }}>
                {/* Glow orbs */}
                <div className="absolute top-[-10%] left-[-10%] w-[80%] h-[80%] rounded-full opacity-20 blur-[120px]" style={{ background: '#415a77' }} />
                <div className="absolute bottom-[-10%] right-[-10%] w-[60%] h-[60%] rounded-full opacity-15 blur-[100px]" style={{ background: '#778da9' }} />

                {/* Grid overlay */}
                <div className="absolute inset-0 opacity-[0.03]"
                    style={{
                        backgroundImage: `linear-gradient(rgba(224,225,221,1) 1px, transparent 1px), linear-gradient(90deg, rgba(224,225,221,1) 1px, transparent 1px)`,
                        backgroundSize: '60px 60px',
                    }}
                />

                <div className="relative z-10 flex flex-col justify-between h-full p-14">
                    <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.6 }}
                        className="flex items-center gap-2.5">
                        <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{ background: 'linear-gradient(135deg, #415a77, #778da9)' }}>
                            <ShieldCheck className="w-5 h-5 text-white" />
                        </div>
                        <span className="font-display font-bold text-lg text-white">PolicyAI</span>
                    </motion.div>

                    <div className="max-w-md">
                        <motion.h1 initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, delay: 0.2 }}
                            className="font-display font-bold text-[3.25rem] leading-[1.05] mb-5" style={{ color: '#e0e1dd' }}>
                            AI-Powered{' '}
                            <span style={{ background: 'linear-gradient(135deg, #778da9, #e0e1dd)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                                Policy Analysis
                            </span>
                        </motion.h1>

                        <motion.p initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, delay: 0.3 }}
                            className="text-base mb-10 leading-relaxed" style={{ color: '#778da9' }}>
                            Extract clauses, detect risks, and map GDPR/CCPA compliance using state-of-the-art NLP models.
                        </motion.p>

                        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }} className="flex flex-col gap-3.5">
                            {[
                                { icon: Brain, text: 'Context-aware Clause Extraction' },
                                { icon: Lock, text: 'Regulatory Compliance Mapping' },
                                { icon: FileText, text: 'Plain-English Summarization' },
                            ].map((f, i) => (
                                <div key={i} className="flex items-center gap-3">
                                    <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: 'rgba(65,90,119,0.2)', border: '1px solid rgba(65,90,119,0.15)' }}>
                                        <f.icon className="w-4 h-4" style={{ color: '#778da9' }} />
                                    </div>
                                    <span className="text-sm font-medium" style={{ color: '#e0e1dd' }}>{f.text}</span>
                                </div>
                            ))}
                        </motion.div>
                    </div>

                    <p className="text-xs" style={{ color: '#415a77' }}>© 2026 Intelligent Policy Analyzer</p>
                </div>
            </div>

            {/* ── Right Panel: Form ────────────────────── */}
            <div className="w-full lg:w-1/2 flex items-center justify-center relative" style={{ background: '#e0e1dd' }}>
                {/* Mobile logo */}
                <div className="absolute top-6 left-6 lg:hidden flex items-center gap-2">
                    <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: 'linear-gradient(135deg, #415a77, #778da9)' }}>
                        <ShieldCheck className="w-4 h-4 text-white" />
                    </div>
                    <span className="font-display font-bold" style={{ color: '#0d1b2a' }}>PolicyAI</span>
                </div>

                <div className="w-full max-w-md px-8 py-12">
                    <AnimatePresence mode="wait">
                        <motion.div key={isRegister ? 'reg' : 'login'}
                            initial={{ opacity: 0, x: 10 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -10 }} transition={{ duration: 0.25 }}>

                            <div className="text-center mb-10">
                                <h2 className="font-display font-bold text-[2rem] mb-2" style={{ color: '#0d1b2a' }}>
                                    {isRegister ? 'Create an account' : 'Welcome back'}
                                </h2>
                                <p className="font-medium text-sm" style={{ color: '#778da9' }}>
                                    {isRegister ? 'Start analyzing policies today.' : 'Enter your credentials to continue.'}
                                </p>
                            </div>

                            <form onSubmit={handleSubmit} className="space-y-4">
                                {isRegister && (
                                    <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} className="space-y-4">
                                        <div>
                                            <label className="block text-xs font-medium mb-1.5 ml-0.5" style={{ color: '#415a77' }}>Full Name</label>
                                            <input type="text" placeholder="John Doe"
                                                value={form.full_name}
                                                onChange={e => setForm({ ...form, full_name: e.target.value })}
                                                className={inputClass}
                                                style={{ background: 'white', borderColor: '#d1d5db', color: '#0d1b2a' }}
                                                onFocus={e => e.target.style.borderColor = '#415a77'}
                                                onBlur={e => e.target.style.borderColor = '#d1d5db'}
                                            />
                                        </div>
                                        <div>
                                            <label className="block text-xs font-medium mb-1.5 ml-0.5" style={{ color: '#415a77' }}>Email <span className="text-red-500">*</span></label>
                                            <input type="email" placeholder="john@example.com"
                                                value={form.email}
                                                onChange={e => setForm({ ...form, email: e.target.value })}
                                                className={inputClass} required
                                                style={{ background: 'white', borderColor: '#d1d5db', color: '#0d1b2a' }}
                                                onFocus={e => e.target.style.borderColor = '#415a77'}
                                                onBlur={e => e.target.style.borderColor = '#d1d5db'}
                                            />
                                        </div>
                                    </motion.div>
                                )}

                                <div>
                                    <label className="block text-xs font-medium mb-1.5 ml-0.5" style={{ color: '#415a77' }}>Username <span className="text-red-500">*</span></label>
                                    <input type="text" placeholder="Enter your username"
                                        value={form.username}
                                        onChange={e => setForm({ ...form, username: e.target.value })}
                                        className={inputClass} required
                                        style={{ background: 'white', borderColor: '#d1d5db', color: '#0d1b2a' }}
                                        onFocus={e => e.target.style.borderColor = '#415a77'}
                                        onBlur={e => e.target.style.borderColor = '#d1d5db'}
                                    />
                                </div>

                                <div>
                                    <label className="block text-xs font-medium mb-1.5 ml-0.5" style={{ color: '#415a77' }}>Password <span className="text-red-500">*</span></label>
                                    <input type="password" placeholder="••••••••"
                                        value={form.password}
                                        onChange={e => setForm({ ...form, password: e.target.value })}
                                        className={inputClass} required
                                        style={{ background: 'white', borderColor: '#d1d5db', color: '#0d1b2a' }}
                                        onFocus={e => e.target.style.borderColor = '#415a77'}
                                        onBlur={e => e.target.style.borderColor = '#d1d5db'}
                                    />
                                </div>

                                {error && (
                                    <motion.div initial={{ opacity: 0, y: -5 }} animate={{ opacity: 1, y: 0 }}
                                        className="text-sm p-3 rounded-xl border flex items-start gap-2"
                                        style={{ background: '#fef2f2', borderColor: '#fecaca', color: '#b91c1c' }}>
                                        <span>⚠️</span><span>{error}</span>
                                    </motion.div>
                                )}

                                <button type="submit" disabled={loading}
                                    className="w-full py-3.5 rounded-xl font-semibold text-sm text-white transition-all hover:scale-[1.01] disabled:opacity-60 flex items-center justify-center gap-2 mt-6"
                                    style={{ background: '#0d1b2a', boxShadow: '0 4px 14px rgba(13,27,42,0.3)' }}>
                                    {loading ? (
                                        <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                    ) : (
                                        <>
                                            {isRegister ? 'Create Account' : 'Sign In'}
                                            <ArrowRight className="w-4 h-4 opacity-70" />
                                        </>
                                    )}
                                </button>
                            </form>

                            <div className="mt-8 text-center pt-6" style={{ borderTop: '1px solid #d1d5db' }}>
                                <p className="text-sm" style={{ color: '#778da9' }}>
                                    {isRegister ? 'Already have an account? ' : "Don't have an account? "}
                                    <button
                                        onClick={() => { setIsRegister(!isRegister); setError(''); setForm({ username: '', password: '', email: '', full_name: '' }); }}
                                        className="font-semibold transition-colors" style={{ color: '#415a77' }}>
                                        {isRegister ? 'Sign In' : 'Sign Up'}
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
