import React from 'react';
import { Link } from 'react-router-dom';

const Footer: React.FC = () => {
    return (
        <footer className="border-t border-white/10 bg-[#04040a] relative z-10 w-full">
            <div className="mx-auto max-w-7xl px-6 py-16 lg:px-8">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-12 md:gap-8">
                    <div className="col-span-2 md:col-span-1">
                        <div className="flex items-center gap-2 mb-4">
                            <span className="text-xl font-bold bg-gradient-to-r from-accent-1 to-accent-3 bg-clip-text text-transparent">◎</span>
                            <span className="font-semibold text-text-primary text-lg">VortexML</span>
                        </div>
                        <p className="text-sm text-text-secondary leading-relaxed max-w-xs">
                            The future of machine learning development. Build, configure, and train models without extreme complexity.
                        </p>
                    </div>
                    <div>
                        <h3 className="text-sm font-semibold text-text-primary mb-4 tracking-wider uppercase">Product</h3>
                        <ul className="space-y-3">
                            <li><Link to="/architect" className="text-sm text-text-secondary hover:text-white transition-colors duration-300">Architect Builder</Link></li>
                            <li><Link to="/dataset" className="text-sm text-text-secondary hover:text-white transition-colors duration-300">Dataset Processing</Link></li>
                            <li><Link to="/training" className="text-sm text-text-secondary hover:text-white transition-colors duration-300">Live Training</Link></li>
                        </ul>
                    </div>
                    <div>
                        <h3 className="text-sm font-semibold text-text-primary mb-4 tracking-wider uppercase">Resources</h3>
                        <ul className="space-y-3">
                            <li><Link to="/courses" className="text-sm text-text-secondary hover:text-white transition-colors duration-300">ML Courses</Link></li>
                            <li><a href="#" className="text-sm text-text-secondary hover:text-white transition-colors duration-300">Documentation</a></li>
                            <li><a href="#" className="text-sm text-text-secondary hover:text-white transition-colors duration-300">Community</a></li>
                        </ul>
                    </div>
                    <div>
                        <h3 className="text-sm font-semibold text-text-primary mb-4 tracking-wider uppercase">Company</h3>
                        <ul className="space-y-3">
                            <li><a href="#" className="text-sm text-text-secondary hover:text-white transition-colors duration-300">About Us</a></li>
                            <li><a href="#" className="text-sm text-text-secondary hover:text-white transition-colors duration-300">Blog</a></li>
                            <li><a href="#" className="text-sm text-text-secondary hover:text-white transition-colors duration-300">Contact</a></li>
                        </ul>
                    </div>
                </div>
                <div className="mt-16 pt-8 border-t border-white/5 flex flex-col md:flex-row justify-between items-center gap-4">
                    <p className="text-xs text-text-secondary">&copy; {new Date().getFullYear()} VortexML Platform. All rights reserved.</p>
                    <div className="flex gap-6">
                        <a href="#" className="text-xs text-text-secondary hover:text-white transition-colors">Privacy Policy</a>
                        <a href="#" className="text-xs text-text-secondary hover:text-white transition-colors">Terms of Service</a>
                    </div>
                </div>
            </div>
        </footer>
    );
};

export default Footer;
