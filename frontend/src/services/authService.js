/**
 * Auth Service — wraps authAPI calls with typed responses.
 */
import { authAPI } from './api';

export const authService = {
    login: async (username, password) => {
        const res = await authAPI.login({ username, password });
        return res.data;
    },
    register: async (data) => {
        const res = await authAPI.register(data);
        return res.data;
    },
    getMe: async () => {
        const res = await authAPI.getMe();
        return res.data;
    },
};
