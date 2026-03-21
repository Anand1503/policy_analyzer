import React from 'react';
import { useNavigate } from 'react-router-dom';
import { ShieldCheck, Brain, FileText, Lock, ArrowRight, Sparkles, BarChart3 } from 'lucide-react';
import { motion } from 'framer-motion';

const features = [
    {
        icon: Brain,
        title: 'AI-Powered Analysis',
        desc: 'Deep NLP models extract clauses, detect risks, and classify policy language automatically.',
    },
    {
        icon: Lock,
        title: 'Compliance Mapping',
        desc: 'Maps policies against GDPR, CCPA, and other regulatory frameworks in seconds.',
    },
    {
        icon: BarChart3,
        title: 'Risk Scoring',
        desc: 'Quantitative risk scores with severity levels for every clause in your document.',
    },
    {
        icon: FileText,
        title: 'Smart Summarization',
        desc: 'Get plain-English summaries of complex legal jargon with actionable insights.',
    },
];

const fadeUp = (delay = 0) => ({
    initial: { opacity: 0, y: 30 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.7, delay, ease: [0.22, 1, 0.36, 1] },
});

const LandingPage = () => {
    const navigate = useNavigate();

    return (
        <div className="min-h-screen overflow-hidden" style={{ background: '#0d1b2a' }}>

            {/* ── Nav ──────────────────────────────────────── */}
            <nav className="relative z-20 max-w-7xl mx-auto px-6 py-5 flex items-center justify-between">
                <div className="flex items-center gap-2.5">
                    <div className="w-9 h-9 rounded-xl flex items-center justify-center" style={{ background: 'linear-gradient(135deg, #415a77, #778da9)' }}>
                        <ShieldCheck className="w-5 h-5 text-white" />
                    </div>
                    <span className="font-display font-bold text-lg text-white tracking-tight">PolicyAI</span>
                </div>
                <div className="flex items-center gap-3">
                    <button
                        onClick={() => navigate('/login')}
                        className="text-sm font-medium px-4 py-2 rounded-lg transition-all"
                        style={{ color: '#e0e1dd' }}
                        onMouseEnter={e => e.target.style.color = '#fff'}
                        onMouseLeave={e => e.target.style.color = '#e0e1dd'}
                    >
                        Sign In
                    </button>
                    <button
                        onClick={() => navigate('/login')}
                        className="text-sm font-medium px-5 py-2.5 rounded-xl text-white transition-all hover:scale-[1.02]"
                        style={{ background: 'linear-gradient(135deg, #415a77, #778da9)', boxShadow: '0 4px 20px rgba(65,90,119,0.3)' }}
                    >
                        Get Started
                    </button>
                </div>
            </nav>

            {/* ── Hero ─────────────────────────────────────── */}
            <section className="relative z-10 max-w-7xl mx-auto px-6 pt-20 pb-32">
                {/* Background gradient orbs */}
                <div className="absolute top-0 left-1/4 w-[500px] h-[500px] rounded-full opacity-20 blur-[120px]" style={{ background: '#415a77' }} />
                <div className="absolute top-32 right-1/4 w-[400px] h-[400px] rounded-full opacity-15 blur-[100px]" style={{ background: '#778da9' }} />

                {/* Grid pattern */}
                <div className="absolute inset-0 opacity-[0.03]"
                    style={{
                        backgroundImage: `linear-gradient(rgba(224,225,221,1) 1px, transparent 1px), linear-gradient(90deg, rgba(224,225,221,1) 1px, transparent 1px)`,
                        backgroundSize: '60px 60px',
                    }}
                />

                <div className="relative text-center max-w-4xl mx-auto">
                    {/* Badge */}
                    <motion.div {...fadeUp(0)} className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-medium mb-8"
                        style={{ background: 'rgba(65,90,119,0.2)', color: '#778da9', border: '1px solid rgba(119,141,169,0.2)' }}>
                        <Sparkles className="w-3.5 h-3.5" />
                        AI-Powered Privacy Intelligence Platform
                    </motion.div>

                    {/* Title */}
                    <motion.h1 {...fadeUp(0.1)} className="font-display font-bold text-5xl md:text-7xl leading-[1.05] mb-6" style={{ color: '#e0e1dd' }}>
                        Analyze Policies{' '}
                        <br className="hidden md:block" />
                        <span style={{ background: 'linear-gradient(135deg, #778da9, #e0e1dd)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                            with Intelligence
                        </span>
                    </motion.h1>

                    {/* Subtitle */}
                    <motion.p {...fadeUp(0.2)} className="text-lg md:text-xl max-w-2xl mx-auto mb-10 leading-relaxed" style={{ color: '#778da9' }}>
                        Upload any privacy policy or legal document. Our AI extracts clauses, maps regulatory compliance, scores risks, and generates actionable insights — all automatically.
                    </motion.p>

                    {/* CTA Buttons */}
                    <motion.div {...fadeUp(0.3)} className="flex flex-col sm:flex-row items-center justify-center gap-4">
                        <button
                            onClick={() => navigate('/login')}
                            className="group flex items-center gap-2 px-8 py-4 rounded-xl text-white font-semibold text-base transition-all hover:scale-[1.02]"
                            style={{ background: 'linear-gradient(135deg, #415a77, #778da9)', boxShadow: '0 8px 30px rgba(65,90,119,0.4)' }}
                        >
                            Start Analyzing
                            <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                        </button>
                        <button
                            onClick={() => document.getElementById('features')?.scrollIntoView({ behavior: 'smooth' })}
                            className="flex items-center gap-2 px-8 py-4 rounded-xl font-medium text-base transition-all"
                            style={{ color: '#778da9', border: '1px solid rgba(119,141,169,0.3)' }}
                        >
                            Learn More
                        </button>
                    </motion.div>
                </div>
            </section>

            {/* ── Features ─────────────────────────────────── */}
            <section id="features" className="relative z-10 py-24" style={{ background: '#1b263b' }}>
                <div className="max-w-6xl mx-auto px-6">
                    <motion.div {...fadeUp(0)} className="text-center mb-16">
                        <h2 className="font-display font-bold text-3xl md:text-4xl mb-4" style={{ color: '#e0e1dd' }}>
                            Everything you need
                        </h2>
                        <p className="text-base max-w-lg mx-auto" style={{ color: '#778da9' }}>
                            From upload to actionable intelligence — a complete policy analysis pipeline.
                        </p>
                    </motion.div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                        {features.map((feat, i) => (
                            <motion.div
                                key={feat.title}
                                {...fadeUp(0.1 * i)}
                                className="group p-6 rounded-2xl border transition-all hover:-translate-y-1"
                                style={{
                                    background: 'rgba(65,90,119,0.08)',
                                    borderColor: 'rgba(65,90,119,0.15)',
                                }}
                                onMouseEnter={e => e.currentTarget.style.borderColor = 'rgba(119,141,169,0.3)'}
                                onMouseLeave={e => e.currentTarget.style.borderColor = 'rgba(65,90,119,0.15)'}
                            >
                                <div className="w-10 h-10 rounded-xl flex items-center justify-center mb-4"
                                    style={{ background: 'rgba(65,90,119,0.2)' }}>
                                    <feat.icon className="w-5 h-5" style={{ color: '#778da9' }} />
                                </div>
                                <h3 className="font-display font-semibold text-lg mb-2" style={{ color: '#e0e1dd' }}>{feat.title}</h3>
                                <p className="text-sm leading-relaxed" style={{ color: '#778da9' }}>{feat.desc}</p>
                            </motion.div>
                        ))}
                    </div>
                </div>
            </section>

            {/* ── CTA ──────────────────────────────────────── */}
            <section className="relative z-10 py-24" style={{ background: '#0d1b2a' }}>
                <div className="max-w-3xl mx-auto px-6 text-center">
                    <motion.h2 {...fadeUp(0)} className="font-display font-bold text-3xl md:text-4xl mb-4" style={{ color: '#e0e1dd' }}>
                        Ready to get started?
                    </motion.h2>
                    <motion.p {...fadeUp(0.1)} className="text-base mb-8" style={{ color: '#778da9' }}>
                        Upload your first document and see the results in minutes.
                    </motion.p>
                    <motion.div {...fadeUp(0.2)}>
                        <button
                            onClick={() => navigate('/login')}
                            className="group inline-flex items-center gap-2 px-8 py-4 rounded-xl text-white font-semibold transition-all hover:scale-[1.02]"
                            style={{ background: 'linear-gradient(135deg, #415a77, #778da9)', boxShadow: '0 8px 30px rgba(65,90,119,0.4)' }}
                        >
                            Create Free Account
                            <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                        </button>
                    </motion.div>
                </div>
            </section>

            {/* ── Footer ───────────────────────────────────── */}
            <footer className="relative z-10 py-8 border-t" style={{ background: '#0d1b2a', borderColor: 'rgba(65,90,119,0.2)' }}>
                <div className="max-w-6xl mx-auto px-6 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <ShieldCheck className="w-4 h-4" style={{ color: '#415a77' }} />
                        <span className="text-xs font-medium" style={{ color: '#778da9' }}>PolicyAI</span>
                    </div>
                    <p className="text-xs" style={{ color: '#415a77' }}>© 2026 Intelligent Policy Analyzer. All rights reserved.</p>
                </div>
            </footer>
        </div>
    );
};

export default LandingPage;
