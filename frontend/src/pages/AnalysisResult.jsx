import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { analysisAPI } from '../services/api';
import AnalysisResults from '../components/AnalysisResults';
import { BarChart3, Loader2 } from 'lucide-react';

const AnalysisResult = () => {
    const [searchParams] = useSearchParams();
    const docId = searchParams.get('docId');
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    useEffect(() => {
        if (docId) {
            fetchResults(docId);
        }
    }, [docId]);

    const fetchResults = async (id) => {
        setLoading(true);
        setError('');
        try {
            const res = await analysisAPI.results(id);
            setData(res.data);
        } catch (err) {
            setError(err.response?.data?.detail || 'Failed to fetch results');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="max-w-7xl mx-auto px-4 py-8">
            {loading && (
                <div className="text-center py-20">
                    <Loader2 className="w-10 h-10 text-blue-600 animate-spin mx-auto mb-4" />
                    <p className="text-slate-500">Loading analysis results...</p>
                </div>
            )}
            {error && (
                <div className="bg-red-50 border border-red-200 rounded-xl p-6 text-center text-red-700">
                    {error}
                </div>
            )}
            {data ? (
                <AnalysisResults data={data} />
            ) : !loading && !error && (
                <div className="text-center py-20 text-slate-400">
                    <BarChart3 className="w-12 h-12 mx-auto mb-4 opacity-50" />
                    <p>No analysis results yet. Select a document and click "Analyze".</p>
                </div>
            )}
        </div>
    );
};

export default AnalysisResult;
