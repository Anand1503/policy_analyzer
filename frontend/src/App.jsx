import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import { ThemeProvider } from './context/ThemeContext';
import LandingPage from './pages/LandingPage';
import LoginPage from './pages/LoginPage';
import DashboardPage from './pages/DashboardPage';
import GlobalChatPage from './pages/GlobalChatPage';
import Navbar from './components/Navbar';
import { Loader2 } from 'lucide-react';

const SIDEBAR_WIDTH = '240px';

const ProtectedRoute = ({ children }) => {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'var(--color-alabaster-light)' }}>
        <Loader2 className="animate-spin" style={{ width: '32px', height: '32px', color: '#415a77' }} />
      </div>
    );
  }

  if (!user) return <Navigate to="/login" replace />;

  return (
    <div style={{ display: 'flex', minHeight: '100vh', background: 'var(--color-alabaster-light)', transition: 'background 0.3s' }}>
      {/* Sidebar */}
      <Navbar />

      {/* Main content offset by sidebar width */}
      <main style={{ marginLeft: SIDEBAR_WIDTH, flex: 1, minHeight: '100vh', padding: '24px 32px', }}>
        {children}
      </main>
    </div>
  );
};

const AppRoutes = () => {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'var(--color-alabaster-light)' }}>
        <Loader2 className="animate-spin" style={{ width: '32px', height: '32px', color: '#415a77' }} />
      </div>
    );
  }

  return (
    <Routes>
      {/* Public routes */}
      <Route path="/welcome" element={user ? <Navigate to="/upload" replace /> : <LandingPage />} />
      <Route path="/login" element={user ? <Navigate to="/upload" replace /> : <LoginPage />} />

      {/* Protected routes — upload is the first page after login */}
      <Route path="/upload" element={<ProtectedRoute><DashboardPage initialView="upload" /></ProtectedRoute>} />
      <Route path="/dashboard" element={<ProtectedRoute><DashboardPage initialView="overview" /></ProtectedRoute>} />
      <Route path="/history" element={<ProtectedRoute><DashboardPage initialView="history" /></ProtectedRoute>} />
      <Route path="/chat" element={<ProtectedRoute><GlobalChatPage /></ProtectedRoute>} />

      {/* Default: login if not logged in, upload if logged in */}
      <Route path="/" element={user ? <Navigate to="/upload" replace /> : <Navigate to="/login" replace />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
};

function App() {
  return (
    <BrowserRouter>
      <ThemeProvider>
        <AuthProvider>
          <AppRoutes />
        </AuthProvider>
      </ThemeProvider>
    </BrowserRouter>
  );
}

export default App;
