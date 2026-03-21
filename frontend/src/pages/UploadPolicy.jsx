import React from 'react';
import UploadForm from '../components/UploadForm';
import { useNavigate } from 'react-router-dom';

const UploadPolicy = () => {
    const navigate = useNavigate();

    const handleComplete = (docId) => {
        navigate('/documents');
    };

    return (
        <div className="max-w-7xl mx-auto px-4 py-8">
            <UploadForm onComplete={handleComplete} />
        </div>
    );
};

export default UploadPolicy;
