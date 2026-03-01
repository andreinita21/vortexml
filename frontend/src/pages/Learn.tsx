import React, { useState } from 'react';
import { motion } from 'framer-motion';
import Courses from './Courses';
import Roadmap from './Roadmap';

type LearnTab = 'roadmap' | 'courses';

const tabs: { id: LearnTab; label: string; icon: string }[] = [
    { id: 'roadmap', label: 'Roadmap', icon: '🗺️' },
    { id: 'courses', label: 'Courses', icon: '📚' },
];

const Learn: React.FC = () => {
    const [activeTab, setActiveTab] = useState<LearnTab>('roadmap');

    return (
        <div className="learn-page">
            {/* ── Tab Bar ── */}
            <div className="learn-tab-bar">
                {tabs.map((tab) => (
                    <button
                        key={tab.id}
                        className={`learn-tab ${activeTab === tab.id ? 'active' : ''}`}
                        onClick={() => setActiveTab(tab.id)}
                    >
                        {activeTab === tab.id && (
                            <motion.div
                                className="learn-tab-bg"
                                layoutId="learn-tab-indicator"
                                transition={{
                                    type: 'spring',
                                    stiffness: 400,
                                    damping: 30,
                                }}
                            />
                        )}
                        <span className="learn-tab-icon">{tab.icon}</span>
                        <span className="learn-tab-label">{tab.label}</span>
                    </button>
                ))}
            </div>

            {/* ── Tab Content ── */}
            <div className="learn-content">
                {activeTab === 'courses' ? <Courses /> : <Roadmap />}
            </div>
        </div>
    );
};

export default Learn;
