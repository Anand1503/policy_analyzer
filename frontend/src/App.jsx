import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import LandingPage from './pages/LandingPage';
import LoginPage from './pages/LoginPage';
import DashboardPage from './pages/DashboardPage';
import GlobalChatPage from './pages/GlobalChatPage';
import Navbar from './components/Navbar';
import { Loader2 } from 'lucide-react';

const ProtectedRoute = ({ children }) => {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ background: '#e0e1dd' }}>
        <Loader2 className="w-6 h-6 animate-spin" style={{ color: '#415a77' }} />
      </div>
    );
  }

  if (!user) return <Navigate to="/login" replace />;

  return (
    <div className="min-h-screen" style={{ background: '#edeee9' }}>
      <Navbar />
      {children}
    </div>
  );
};

const AppRoutes = () => {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ background: '#e0e1dd' }}>
        <Loader2 className="w-6 h-6 animate-spin" style={{ color: '#415a77' }} />
      </div>
    );
  }

  return (
    <Routes>
      {/* Public routes */}
      <Route path="/welcome" element={user ? <Navigate to="/dashboard" replace /> : <LandingPage />} />
      <Route path="/login" element={user ? <Navigate to="/dashboard" replace /> : <LoginPage />} />

      {/* Protected routes */}
      <Route path="/dashboard" element={<ProtectedRoute><DashboardPage /></ProtectedRoute>} />
      <Route path="/upload" element={<ProtectedRoute><DashboardPage initialView="upload" /></ProtectedRoute>} />
      <Route path="/history" element={<ProtectedRoute><DashboardPage initialView="history" /></ProtectedRoute>} />
      <Route path="/chat" element={<ProtectedRoute><GlobalChatPage /></ProtectedRoute>} />

      {/* Default: landing if not logged in, dashboard if logged in */}
      <Route path="/" element={user ? <Navigate to="/dashboard" replace /> : <Navigate to="/welcome" replace />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
};

function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <AppRoutes />
      </AuthProvider>
    </BrowserRouter>
  );
}

export default App;
