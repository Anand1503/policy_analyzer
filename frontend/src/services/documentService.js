/**
 * Document Service — wraps docsAPI calls.
 */
import { docsAPI } from './api';

export const documentService = {
    upload: async (file, onProgress) => {
        const res = await docsAPI.upload(file, onProgress);
        return res.data;
    },
    list: async () => {
        const res = await docsAPI.list();
        return res.data;
    },
    get: async (id) => {
        const res = await docsAPI.get(id);
        return res.data;
    },
    delete: async (id) => {
        const res = await docsAPI.delete(id);
        return res.data;
    },
};
