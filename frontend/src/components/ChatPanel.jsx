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
        } catch (err) {
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
        <div className="bg-white rounded-xl shadow-sm border border-slate-200 flex flex-col h-[600px]">
            {/* Header */}
            <div className="p-4 border-b border-slate-100 flex items-center gap-3">
                <div className="bg-indigo-100 p-2 rounded-lg">
                    <Brain className="w-5 h-5 text-indigo-600" />
                </div>
                <div>
                    <h3 className="font-bold text-slate-800">AI Policy Assistant</h3>
                    <p className="text-xs text-slate-400">Ask questions about the document</p>
                </div>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {messages.length === 0 && (
                    <div className="text-center py-12">
                        <Brain className="w-12 h-12 text-slate-200 mx-auto mb-4" />
                        <p className="text-slate-400 text-sm mb-6">Ask anything about the uploaded document</p>
                        <div className="grid grid-cols-2 gap-2 max-w-md mx-auto">
                            {SUGGESTIONS.map((s, i) => (
                                <button
                                    key={i}
                                    onClick={() => setInput(s)}
                                    className="text-xs bg-slate-50 hover:bg-indigo-50 border border-slate-200 hover:border-indigo-200 rounded-lg p-3 text-left text-slate-600 hover:text-indigo-700 transition-all"
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
                        className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : ''}`}
                    >
                        {msg.role === 'assistant' && (
                            <div className="bg-indigo-100 p-1.5 rounded-lg h-fit flex-shrink-0">
                                <Brain className="w-4 h-4 text-indigo-600" />
                            </div>
                        )}
                        <div className={`max-w-[80%] rounded-xl px-4 py-3 ${msg.role === 'user'
                                ? 'bg-blue-600 text-white'
                                : msg.error
                                    ? 'bg-red-50 text-red-700 border border-red-200'
                                    : 'bg-slate-100 text-slate-700'
                            }`}>
                            {msg.role === 'assistant' ? (
                                <div className="prose prose-sm max-w-none">
                                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                                </div>
                            ) : (
                                <p className="text-sm">{msg.content}</p>
                            )}

                            {/* Sources */}
                            {msg.sources?.length > 0 && (
                                <div className="mt-3 pt-3 border-t border-slate-200 space-y-2">
                                    <p className="text-xs font-semibold text-slate-500 flex items-center gap-1">
                                        <BookOpen className="w-3 h-3" /> Sources
                                    </p>
                                    {msg.sources.map((src, j) => (
                                        <div key={j} className="text-xs bg-white/70 rounded p-2 text-slate-500 leading-relaxed">
                                            {src.slice(0, 200)}...
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                        {msg.role === 'user' && (
                            <div className="bg-blue-100 p-1.5 rounded-lg h-fit flex-shrink-0">
                                <User className="w-4 h-4 text-blue-600" />
                            </div>
                        )}
                    </motion.div>
                ))}

                {loading && (
                    <div className="flex gap-3">
                        <div className="bg-indigo-100 p-1.5 rounded-lg h-fit">
                            <Brain className="w-4 h-4 text-indigo-600" />
                        </div>
                        <div className="bg-slate-100 rounded-xl px-4 py-3 flex items-center gap-2">
                            <Loader2 className="w-4 h-4 animate-spin text-slate-400" />
                            <span className="text-sm text-slate-400">Thinking...</span>
                        </div>
                    </div>
                )}

                <div ref={scrollRef} />
            </div>

            {/* Input */}
            <form onSubmit={sendMessage} className="p-4 border-t border-slate-100 flex gap-3">
                <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    placeholder="Ask about the policy..."
                    className="flex-1 bg-slate-50 border border-slate-200 rounded-lg px-4 py-2.5 text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
                    disabled={loading}
                />
                <button
                    type="submit"
                    disabled={loading || !input.trim()}
                    className="bg-blue-600 hover:bg-blue-700 text-white px-4 rounded-lg transition-colors disabled:opacity-50"
                >
                    <Send className="w-4 h-4" />
                </button>
            </form>
        </div>
    );
};

export default ChatPanel;
