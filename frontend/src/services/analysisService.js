/**
 * Analysis Service — wraps analysisAPI calls.
 */
import { analysisAPI } from './api';

export const analysisService = {
    trigger: async (docId) => {
        const res = await analysisAPI.trigger(docId);
        return res.data;
    },
    results: async (docId) => {
        const res = await analysisAPI.results(docId);
        return res.data;
    },
    chat: async (query, docId) => {
        const res = await analysisAPI.chat(query, docId);
        return res.data;
    },
};
