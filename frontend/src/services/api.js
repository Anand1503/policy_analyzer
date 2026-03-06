/**
 * API Client — Axios instance with JWT interceptor.
 */

import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

const api = axios.create({ baseURL: API_BASE });

// ─── JWT Interceptor ────────────────────────────────────
api.interceptors.request.use((config) => {
    const token = localStorage.getItem('token');
    if (token) {
        config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
});

api.interceptors.response.use(
    (res) => res,
    (err) => {
        if (err.response?.status === 401) {
            localStorage.removeItem('token');
            window.location.href = '/login';
        }
        return Promise.reject(err);
    }
);

// ─── Auth ───────────────────────────────────────────────
export const authAPI = {
    register: (data) => api.post('/auth/register', data),
    login: (data) => api.post('/auth/login', data),
    getMe: () => api.get('/auth/me'),
};

// ─── Documents ──────────────────────────────────────────
export const docsAPI = {
    upload: (file, onProgress) => {
        const form = new FormData();
        form.append('file', file);
        return api.post('/documents/upload', form, {
            headers: { 'Content-Type': 'multipart/form-data' },
            onUploadProgress: onProgress,
        });
    },
    list: () => api.get('/documents/'),
    get: (id) => api.get(`/documents/${id}`),
    delete: (id) => api.delete(`/documents/${id}`),
};

// ─── Analysis ───────────────────────────────────────────
export const analysisAPI = {
    trigger: (docId) => api.post(`/analysis/analyze/${docId}`),
    results: (docId) => api.get(`/analysis/results/${docId}`),
    chat: (query, docId) => api.post('/analysis/chat', { query, document_id: docId }),
};

export default api;
