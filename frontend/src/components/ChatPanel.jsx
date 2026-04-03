import React, { useState, useRef, useEffect } from 'react';
import { analysisAPI } from '../services/api';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Brain, Send, Loader2, User, BookOpen } from 'lucide-react';
import { motion } from 'framer-motion';

const ChatPanel = ({ documentId }) => {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const scrollRef = useRef(null);

    useEffect(() => {
        scrollRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const sendMessage = async (e) => {
        e.preventDefault();
        if (!input.trim() || loading) return;

        const userMsg = { role: 'user', content: input };
        setMessages((prev) => [...prev, userMsg]);
        setInput('');
        setLoading(true);

        try {
            const res = await analysisAPI.chat(input, documentId);
            setMessages((prev) => [
                ...prev,
                {
                    role: 'assistant',
                    content: res.data.answer,
                    sources: res.data.sources || [],
                },
            ]);
        } catch {
            setMessages((prev) => [
                ...prev,
                { role: 'assistant', content: 'Sorry, something went wrong. Please try again.', error: true },
            ]);
        } finally {
            setLoading(false);
        }
    };

    const SUGGESTIONS = [
        "What data does this policy collect?",
        "Can my data be sold to third parties?",
        "What are my rights to delete my data?",
        "Is there a data retention period?",
    ];

    return (
        <div style={{
            background: 'var(--color-card)',
            borderRadius: '16px',
            border: '1px solid var(--color-card-border)',
            boxShadow: 'var(--shadow-soft)',
            display: 'flex',
            flexDirection: 'column',
            height: '600px',
            overflow: 'hidden',
        }}>
            {/* Header */}
            <div style={{
                padding: '16px 20px',
                borderBottom: '1px solid var(--color-card-border)',
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
                background: 'rgba(99,102,241,0.04)',
            }}>
                <div style={{
                    background: 'rgba(99,102,241,0.12)',
                    padding: '8px',
                    borderRadius: '10px',
                }}>
                    <Brain size={20} style={{ color: 'var(--color-accent)' }} />
                </div>
                <div>
                    <h3 style={{ fontWeight: '700', color: 'var(--color-text-primary)', margin: 0, fontSize: '15px' }}>AI Policy Assistant</h3>
                    <p style={{ fontSize: '12px', color: 'var(--color-text-muted)', margin: 0 }}>Ask questions about the document</p>
                </div>
            </div>

            {/* Messages */}
            <div style={{
                flex: 1,
                overflowY: 'auto',
                padding: '20px',
                display: 'flex',
                flexDirection: 'column',
                gap: '16px',
            }}>
                {messages.length === 0 && (
                    <div style={{ textAlign: 'center', padding: '48px 0' }}>
                        <Brain size={48} style={{ color: 'var(--color-card-border)', margin: '0 auto 16px' }} />
                        <p style={{ color: 'var(--color-text-muted)', fontSize: '14px', marginBottom: '24px' }}>
                            Ask anything about the uploaded document
                        </p>
                        <div style={{
                            display: 'grid',
                            gridTemplateColumns: '1fr 1fr',
                            gap: '8px',
                            maxWidth: '420px',
                            margin: '0 auto',
                        }}>
                            {SUGGESTIONS.map((s, i) => (
                                <button
                                    key={i}
                                    onClick={() => setInput(s)}
                                    style={{
                                        fontSize: '12px',
                                        background: 'rgba(99,102,241,0.04)',
                                        border: '1px solid var(--color-card-border)',
                                        borderRadius: '10px',
                                        padding: '12px',
                                        textAlign: 'left',
                                        color: 'var(--color-text-primary)',
                                        cursor: 'pointer',
                                        transition: 'all 0.2s',
                                    }}
                                    onMouseEnter={e => {
                                        e.currentTarget.style.background = 'rgba(99,102,241,0.08)';
                                        e.currentTarget.style.borderColor = 'rgba(99,102,241,0.3)';
                                    }}
                                    onMouseLeave={e => {
                                        e.currentTarget.style.background = 'rgba(99,102,241,0.04)';
                                        e.currentTarget.style.borderColor = 'var(--color-card-border)';
                                    }}
                                >
                                    {s}
                                </button>
                            ))}
                        </div>
                    </div>
                )}

                {messages.map((msg, i) => (
                    <motion.div
                        key={i}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        style={{
                            display: 'flex',
                            gap: '10px',
                            justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start',
                        }}
                    >
                        {msg.role === 'assistant' && (
                            <div style={{
                                background: 'rgba(99,102,241,0.12)',
                                padding: '6px',
                                borderRadius: '8px',
                                height: 'fit-content',
                                flexShrink: 0,
                            }}>
                                <Brain size={16} style={{ color: 'var(--color-accent)' }} />
                            </div>
                        )}
                        <div style={{
                            maxWidth: '80%',
                            borderRadius: '14px',
                            padding: '12px 16px',
                            ...(msg.role === 'user'
                                ? {
                                    background: 'linear-gradient(135deg, var(--color-accent), var(--color-accent-light))',
                                    color: '#ffffff',
                                }
                                : msg.error
                                    ? {
                                        background: 'rgba(239,68,68,0.08)',
                                        color: 'var(--color-danger)',
                                        border: '1px solid rgba(239,68,68,0.2)',
                                    }
                                    : {
                                        background: 'var(--color-input-bg)',
                                        color: 'var(--color-text-primary)',
                                        border: '1px solid var(--color-card-border)',
                                    }
                            ),
                        }}>
                            {msg.role === 'assistant' ? (
                                <div style={{ fontSize: '14px', lineHeight: '1.6' }}>
                                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                                </div>
                            ) : (
                                <p style={{ fontSize: '14px', margin: 0, lineHeight: '1.5' }}>{msg.content}</p>
                            )}

                            {/* Sources */}
                            {msg.sources?.length > 0 && (
                                <div style={{
                                    marginTop: '12px',
                                    paddingTop: '12px',
                                    borderTop: '1px solid var(--color-card-border)',
                                }}>
                                    <p style={{
                                        fontSize: '11px',
                                        fontWeight: '600',
                                        color: 'var(--color-text-muted)',
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: '4px',
                                        marginBottom: '8px',
                                    }}>
                                        <BookOpen size={12} /> Sources
                                    </p>
                                    {msg.sources.map((src, j) => (
                                        <div key={j} style={{
                                            fontSize: '12px',
                                            background: 'rgba(119,141,169,0.06)',
                                            borderRadius: '8px',
                                            padding: '8px',
                                            color: 'var(--color-text-muted)',
                                            lineHeight: '1.4',
                                            marginBottom: '4px',
                                        }}>
                                            {src.slice(0, 200)}...
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                        {msg.role === 'user' && (
                            <div style={{
                                background: 'rgba(99,102,241,0.12)',
                                padding: '6px',
                                borderRadius: '8px',
                                height: 'fit-content',
                                flexShrink: 0,
                            }}>
                                <User size={16} style={{ color: 'var(--color-accent)' }} />
                            </div>
                        )}
                    </motion.div>
                ))}

                {loading && (
                    <div style={{ display: 'flex', gap: '10px' }}>
                        <div style={{
                            background: 'rgba(99,102,241,0.12)',
                            padding: '6px',
                            borderRadius: '8px',
                            height: 'fit-content',
                        }}>
                            <Brain size={16} style={{ color: 'var(--color-accent)' }} />
                        </div>
                        <div style={{
                            background: 'var(--color-input-bg)',
                            borderRadius: '14px',
                            padding: '12px 16px',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '8px',
                            border: '1px solid var(--color-card-border)',
                        }}>
                            <Loader2 size={16} style={{ color: 'var(--color-accent)', animation: 'spin 1s linear infinite' }} />
                            <span style={{ fontSize: '14px', color: 'var(--color-text-muted)' }}>Thinking...</span>
                        </div>
                    </div>
                )}

                <div ref={scrollRef} />
            </div>

            {/* Input */}
            <form onSubmit={sendMessage} style={{
                padding: '16px 20px',
                borderTop: '1px solid var(--color-card-border)',
                display: 'flex',
                gap: '10px',
            }}>
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Ask about the policy..."
                    style={{
                        flex: 1,
                        background: 'var(--color-input-bg)',
                        border: '1.5px solid var(--color-card-border)',
                        borderRadius: '12px',
                        padding: '10px 16px',
                        fontSize: '14px',
                        color: 'var(--color-text-primary)',
                        outline: 'none',
                        transition: 'border-color 0.2s',
                        fontFamily: 'inherit',
                    }}
                    onFocus={e => e.currentTarget.style.borderColor = 'var(--color-accent)'}
                    onBlur={e => e.currentTarget.style.borderColor = 'var(--color-card-border)'}
                    disabled={loading}
                />
                <button
                    type="submit"
                    disabled={loading || !input.trim()}
                    style={{
                        background: input.trim() ? 'linear-gradient(135deg, var(--color-accent), var(--color-accent-light))' : 'var(--color-card-border)',
                        color: input.trim() ? '#ffffff' : 'var(--color-text-muted)',
                        border: 'none',
                        borderRadius: '12px',
                        padding: '0 16px',
                        cursor: input.trim() ? 'pointer' : 'default',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        transition: 'all 0.2s',
                        boxShadow: input.trim() ? '0 2px 10px rgba(99,102,241,0.3)' : 'none',
                    }}
                >
                    <Send size={16} />
                </button>
            </form>
        </div>
    );
};

export default ChatPanel;
