import React from 'react';
import UploadForm from '../components/UploadForm';
import { useNavigate } from 'react-router-dom';

const UploadPolicy = () => {
    const navigate = useNavigate();

    const handleComplete = () => {
        navigate('/documents');
    };

    return (
        <div className="max-w-4xl mx-auto px-4 pb-8 pt-2 h-full min-h-[calc(100vh-80px)] flex flex-col">
            <UploadForm onComplete={handleComplete} />
        </div>
    );
};

export default UploadPolicy;
