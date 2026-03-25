import React, { useState, useRef, useEffect } from 'react';
import { analysisAPI, docsAPI } from '../services/api';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Brain, Send, Loader2, User, BookOpen, FileText, Globe } from 'lucide-react';
import { motion } from 'framer-motion';

const GlobalChatPage = () => {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [documents, setDocuments] = useState([]);
    const scrollRef = useRef(null);

    useEffect(() => {
        docsAPI.list().then(res => setDocuments(res.data?.documents || [])).catch(() => {});
    }, []);

    useEffect(() => {
        scrollRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const sendMessage = async (e) => {
        e.preventDefault();
        if (!input.trim() || loading) return;

        const userMsg = { role: 'user', content: input };
        setMessages(prev => [...prev, userMsg]);
        setInput('');
        setLoading(true);

        try {
            // Global chat — no document_id filter, queries all documents
            const res = await analysisAPI.chat(input, null);
            setMessages(prev => [
                ...prev,
                {
                    role: 'assistant',
                    content: res.data.answer,
                    sources: res.data.sources || [],
                },
            ]);
        } catch (err) {
            setMessages(prev => [
                ...prev,
                { role: 'assistant', content: 'Sorry, something went wrong. Please try again.', error: true },
            ]);
        } finally {
            setLoading(false);
        }
    };

    const SUGGESTIONS = [
        "What do the policies say about data collection?",
        "Which policies allow third-party data sharing?",
        "What retention periods are mentioned?",
        "Are there any GDPR compliance gaps?",
        "What security measures are commonly mentioned?",
        "Do any policies mention children's data?",
    ];

    return (
        <div style={{ padding: '2rem', maxWidth: '900px', margin: '0 auto' }}>
            {/* Header */}
            <div style={{
                display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1.5rem'
            }}>
                <div style={{
                    background: '#1b263b', padding: '0.75rem', borderRadius: '12px',
                    display: 'flex', alignItems: 'center', justifyContent: 'center'
                }}>
                    <Globe style={{ width: '24px', height: '24px', color: '#e0e1dd' }} />
                </div>
                <div>
                    <h1 style={{
                        fontSize: '1.5rem', fontWeight: '700', margin: 0, color: '#0d1b2a'
                    }}>Global Policy Assistant</h1>
                    <p style={{
                        fontSize: '0.85rem', color: '#778da9', margin: 0
                    }}>
                        Ask questions across all {documents.length} uploaded documents
                    </p>
                </div>
            </div>

            {/* Chat Container */}
            <div style={{
                background: '#fff', borderRadius: '16px', border: '1px solid #e0e1dd',
                display: 'flex', flexDirection: 'column', height: 'calc(100vh - 200px)',
                boxShadow: '0 4px 24px rgba(13, 27, 42, 0.08)'
            }}>
                {/* Messages */}
                <div style={{
                    flex: 1, overflowY: 'auto', padding: '1.5rem',
                    display: 'flex', flexDirection: 'column', gap: '1rem'
                }}>
                    {messages.length === 0 && (
                        <div style={{ textAlign: 'center', paddingTop: '3rem' }}>
                            <Brain style={{
                                width: '48px', height: '48px', color: '#778da9',
                                margin: '0 auto 1rem', opacity: 0.4
                            }} />
                            <p style={{ color: '#778da9', fontSize: '0.9rem', marginBottom: '1.5rem' }}>
                                Ask anything about all your uploaded policies
                            </p>
                            <div style={{
                                display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))',
                                gap: '0.5rem', maxWidth: '600px', margin: '0 auto'
                            }}>
                                {SUGGESTIONS.map((s, i) => (
                                    <button
                                        key={i}
                                        onClick={() => setInput(s)}
                                        style={{
                                            background: '#f5f6f4', border: '1px solid #e0e1dd',
                                            borderRadius: '10px', padding: '0.75rem',
                                            fontSize: '0.8rem', color: '#415a77',
                                            cursor: 'pointer', textAlign: 'left',
                                            transition: 'all 0.2s'
                                        }}
                                        onMouseOver={e => {
                                            e.target.style.background = '#1b263b';
                                            e.target.style.color = '#e0e1dd';
                                        }}
                                        onMouseOut={e => {
                                            e.target.style.background = '#f5f6f4';
                                            e.target.style.color = '#415a77';
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
                                display: 'flex', gap: '0.75rem',
                                justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start'
                            }}
                        >
                            {msg.role === 'assistant' && (
                                <div style={{
                                    background: '#1b263b', padding: '6px',
                                    borderRadius: '8px', height: 'fit-content', flexShrink: 0
                                }}>
                                    <Brain style={{ width: '16px', height: '16px', color: '#e0e1dd' }} />
                                </div>
                            )}
                            <div style={{
                                maxWidth: '80%', borderRadius: '14px',
                                padding: '0.75rem 1rem',
                                background: msg.role === 'user' ? '#415a77' :
                                    msg.error ? '#fef2f2' : '#f5f6f4',
                                color: msg.role === 'user' ? '#e0e1dd' :
                                    msg.error ? '#b91c1c' : '#0d1b2a',
                                border: msg.error ? '1px solid #fecaca' : 'none',
                            }}>
                                {msg.role === 'assistant' ? (
                                    <div className="prose prose-sm" style={{ maxWidth: 'none', fontSize: '0.9rem' }}>
                                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                                    </div>
                                ) : (
                                    <p style={{ margin: 0, fontSize: '0.9rem' }}>{msg.content}</p>
                                )}

                                {msg.sources?.length > 0 && (
                                    <div style={{
                                        marginTop: '0.75rem', paddingTop: '0.75rem',
                                        borderTop: '1px solid #e0e1dd'
                                    }}>
                                        <p style={{
                                            fontSize: '0.7rem', fontWeight: 600, color: '#778da9',
                                            display: 'flex', alignItems: 'center', gap: '4px', margin: '0 0 0.5rem'
                                        }}>
                                            <BookOpen style={{ width: '12px', height: '12px' }} /> Sources
                                        </p>
                                        {msg.sources.map((src, j) => (
                                            <div key={j} style={{
                                                fontSize: '0.75rem', background: 'rgba(255,255,255,0.7)',
                                                borderRadius: '6px', padding: '0.5rem',
                                                color: '#778da9', lineHeight: 1.5, marginBottom: '0.25rem'
                                            }}>
                                                {src.slice(0, 200)}...
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                            {msg.role === 'user' && (
                                <div style={{
                                    background: '#415a77', padding: '6px',
                                    borderRadius: '8px', height: 'fit-content', flexShrink: 0
                                }}>
                                    <User style={{ width: '16px', height: '16px', color: '#e0e1dd' }} />
                                </div>
                            )}
                        </motion.div>
                    ))}

                    {loading && (
                        <div style={{ display: 'flex', gap: '0.75rem' }}>
                            <div style={{
                                background: '#1b263b', padding: '6px',
                                borderRadius: '8px', height: 'fit-content'
                            }}>
                                <Brain style={{ width: '16px', height: '16px', color: '#e0e1dd' }} />
                            </div>
                            <div style={{
                                background: '#f5f6f4', borderRadius: '14px',
                                padding: '0.75rem 1rem', display: 'flex',
                                alignItems: 'center', gap: '0.5rem'
                            }}>
                                <Loader2 style={{
                                    width: '16px', height: '16px', color: '#778da9',
                                    animation: 'spin 1s linear infinite'
                                }} />
                                <span style={{ fontSize: '0.85rem', color: '#778da9' }}>
                                    Searching across all documents...
                                </span>
                            </div>
                        </div>
                    )}

                    <div ref={scrollRef} />
                </div>

                {/* Input */}
                <form onSubmit={sendMessage} style={{
                    padding: '1rem 1.5rem', borderTop: '1px solid #e0e1dd',
                    display: 'flex', gap: '0.75rem'
                }}>
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Ask about all your policies..."
                        style={{
                            flex: 1, background: '#f5f6f4', border: '1px solid #e0e1dd',
                            borderRadius: '10px', padding: '0.75rem 1rem',
                            fontSize: '0.9rem', outline: 'none', color: '#0d1b2a'
                        }}
                        onFocus={e => e.target.style.borderColor = '#415a77'}
                        onBlur={e => e.target.style.borderColor = '#e0e1dd'}
                        disabled={loading}
                    />
                    <button
                        type="submit"
                        disabled={loading || !input.trim()}
                        style={{
                            background: '#415a77', color: '#e0e1dd',
                            padding: '0 1.25rem', borderRadius: '10px',
                            border: 'none', cursor: 'pointer',
                            opacity: loading || !input.trim() ? 0.5 : 1,
                            transition: 'all 0.2s'
                        }}
                    >
                        <Send style={{ width: '16px', height: '16px' }} />
                    </button>
                </form>
            </div>
        </div>
    );
};

export default GlobalChatPage;
