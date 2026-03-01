import React, { useRef } from 'react';
import { Link } from 'react-router-dom';
import { motion, useScroll, useTransform } from 'framer-motion';
import { Database, Activity, ArrowRight, Layers, Cpu, Network } from 'lucide-react';
import NeuralBackground from '../components/NeuralBackground';
import { cn } from '../utils/cn';

// --- Shared Reusable Animation Component ---
const FadeIn = ({ children, delay = 0, className, direction = "up" }: { children: React.ReactNode, delay?: number, className?: string, direction?: "up" | "down" | "left" | "right" }) => {
    const directions = {
        up: { y: 40, x: 0 },
        down: { y: -40, x: 0 },
        left: { x: 40, y: 0 },
        right: { x: -40, y: 0 },
    };
    return (
        <motion.div
            initial={{ opacity: 0, ...directions[direction] }}
            whileInView={{ opacity: 1, y: 0, x: 0 }}
            viewport={{ once: true, margin: "-10%" }}
            transition={{ duration: 0.8, delay, ease: [0.21, 0.47, 0.32, 0.98] }}
            className={className}
        >
            {children}
        </motion.div>
    );
};

const GlassCard = ({
    title,
    description,
    icon: Icon,
    to,
    className,
    delay = 0
}: {
    title: string,
    description: string,
    icon: React.ElementType,
    to: string,
    className?: string,
    delay?: number
}) => {
    return (
        <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-10%" }}
            transition={{ duration: 0.8, delay, ease: [0.21, 0.47, 0.32, 0.98] }}
            className={cn(
                "group relative flex flex-col justify-between overflow-hidden rounded-[2rem] p-8",
                "bg-[#0a0a16]/40 backdrop-blur-2xl border border-white/5",
                "hover:border-accent-2/40 hover:bg-[#0f0f24]/60 transition-all duration-700",
                "shadow-[0_8px_32px_0_rgba(0,0,0,0.3)] hover:shadow-[0_8px_40px_rgba(139,92,246,0.15)]",
                className
            )}
        >
            {/* Apple Liquid Glass Gradient Overlay on Hover */}
            <div className="absolute inset-0 bg-gradient-to-br from-white/5 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-700" />

            <div className="relative z-10">
                <div className="mb-6 inline-flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-white/10 to-white/5 border border-white/10 text-white shadow-inner group-hover:scale-110 transition-transform duration-500 ease-out">
                    <Icon className="h-6 w-6" />
                </div>
                <h3 className="mb-3 text-2xl font-bold tracking-tight text-white">{title}</h3>
                <p className="text-text-secondary leading-relaxed font-light">{description}</p>
            </div>

            <div className="relative z-10 mt-10">
                <Link
                    to={to}
                    className="inline-flex items-center gap-2 text-sm font-semibold text-white/70 group-hover:text-white transition-colors duration-300 no-underline hover:no-underline"
                >
                    Explore Module
                    <ArrowRight className="h-4 w-4 transform group-hover:translate-x-1.5 transition-transform duration-300" />
                </Link>
            </div>

            {/* Corner Glow */}
            <div className="absolute -bottom-8 -right-8 w-32 h-32 bg-accent-1/20 blur-3xl rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-700" />
        </motion.div>
    );
};

const ScrollIndicator = () => (
    <motion.div
        className="absolute bottom-10 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2 opacity-50"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.5, duration: 1 }}
    >
        <span className="text-[10px] tracking-widest uppercase text-white font-mono">Scroll</span>
        <div className="w-[1px] h-12 bg-gradient-to-b from-white/50 to-transparent relative overflow-hidden">
            <motion.div
                className="w-full h-1/2 bg-white"
                animate={{ y: ["-100%", "200%"] }}
                transition={{ repeat: Infinity, duration: 1.5, ease: "linear" }}
            />
        </div>
    </motion.div>
);

const Home: React.FC = () => {
    const heroRef = useRef<HTMLDivElement>(null);
    const { scrollYProgress } = useScroll({
        target: heroRef,
        offset: ["start start", "end start"]
    });

    // Translate texts on scroll for parallax
    const yHeroText = useTransform(scrollYProgress, [0, 1], ["0%", "50%"]);
    const opacityHero = useTransform(scrollYProgress, [0, 0.8], [1, 0]);

    return (
        <div className="relative w-full bg-bg-primary overflow-hidden selection:bg-accent-2/30">
            {/* Interactive Canvas Background - Appears behind everything */}
            <NeuralBackground />

            {/* Static Gradient Blobs to augment canvas */}
            <div className="pointer-events-none absolute inset-0 overflow-hidden mix-blend-screen opacity-40">
                <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] rounded-full bg-accent-1/20 blur-[150px]" />
                <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] rounded-full bg-accent-4/10 blur-[150px]" />
            </div>

            {/* Subliminal Grid Pattern Overlay */}
            <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAiIGhlaWdodD0iNDAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHBhdGggZD0iTTAgMGg0MHY0MEgweiIgZmlsbD0ibm9uZSIvPjxwYXRoIGQ9Ik0wIDM5LjVMMDQwIDM5LjUiIHN0cm9rZT0icmdiYSgyNTUsMjU1LDI1NSwwLjAyKSIvPjxwYXRoIGQ9Ik0zOS41IDB2NDAiIHN0cm9rZT0icmdiYSgyNTUsMjU1LDI1NSwwLjAyKSIvPjwvc3ZnPg==')] opacity-[0.15] z-0 pointer-events-none" />

            <main className="relative z-10">

                {/* Hero Viewport (Full Screen) */}
                <section ref={heroRef} className="relative min-h-[calc(100vh-64px)] flex flex-col items-center justify-center px-6 pt-10 pb-24">
                    <motion.div
                        style={{ y: yHeroText, opacity: opacityHero }}
                        className="max-w-5xl mx-auto text-center flex flex-col items-center"
                    >
                        <motion.div
                            initial={{ opacity: 0, scale: 0.9 }}
                            animate={{ opacity: 1, scale: 1 }}
                            transition={{ duration: 0.8, ease: "easeOut" }}
                            className="mb-8 inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-5 py-2 backdrop-blur-md shadow-[0_0_20px_rgba(255,255,255,0.05)]"
                        >
                            <span className="flex h-2 w-2 rounded-full bg-accent-4">
                                <span className="animate-ping absolute inline-flex h-2 w-2 rounded-full bg-accent-4 opacity-75"></span>
                            </span>
                            <span className="text-xs font-semibold tracking-widest text-white/90 uppercase font-mono">Vortex Engine</span>
                        </motion.div>

                        <motion.h1
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.8, delay: 0.1, ease: "easeOut" }}
                            className="text-5xl sm:text-7xl md:text-8xl font-bold tracking-tighter text-white leading-[1.05] mb-8"
                        >
                            Intelligence, <br />
                            <span className="text-transparent bg-clip-text bg-gradient-to-r from-accent-1 via-accent-3 to-accent-4 pb-2 filter drop-shadow-[0_0_20px_rgba(139,92,246,0.3)]">
                                Simplified.
                            </span>
                        </motion.h1>

                        <motion.p
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.8, delay: 0.2, ease: "easeOut" }}
                            className="mx-auto max-w-2xl text-lg md:text-xl text-text-secondary leading-relaxed mb-12 font-light"
                        >
                            The most advanced visual node platform for neural networks.
                            Build, configure, and train models in a gorgeous, highly-optimized environment.
                        </motion.p>

                        <motion.div
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.8, delay: 0.3, ease: "easeOut" }}
                            className="flex flex-col sm:flex-row gap-5 justify-center items-center"
                        >
                            <Link
                                to="/dataset"
                                className="group relative inline-flex h-12 md:h-14 items-center justify-center gap-2 overflow-hidden rounded-full bg-white px-8 font-semibold text-black transition-all hover:scale-[1.02] active:scale-[0.98] shadow-[0_0_30px_rgba(255,255,255,0.2)]"
                            >
                                Start Building
                                <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
                            </Link>
                        </motion.div>
                    </motion.div>

                    <ScrollIndicator />
                </section>

                {/* Features Section (Bento Grid) */}
                <section className="relative z-10 max-w-7xl mx-auto px-6 py-32 lg:px-8">
                    <FadeIn className="mb-16 md:mb-24 text-center max-w-3xl mx-auto">
                        <h2 className="text-3xl md:text-5xl font-bold tracking-tight text-white mb-6">Designed for velocity.</h2>
                        <p className="text-lg text-text-secondary font-light leading-relaxed">
                            Every aspect of the platform is engineered to strip away boilerplate and let you focus on what matters: the architecture.
                        </p>
                    </FadeIn>

                    <div className="grid grid-cols-1 md:grid-cols-6 gap-6 md:auto-rows-[26rem]">

                        <GlassCard
                            title="Dataset Engine"
                            description="Streamline data ingestion. Automatically detect features, target columns, and handle complex preprocessing directly within a highly responsive visual workspace."
                            icon={Database}
                            to="/dataset"
                            delay={0.1}
                            className="md:col-span-4"
                        />

                        <GlassCard
                            title="Visual Architecture"
                            description="Drag, drop, and connect neural layers. Visualize deep networks instantly."
                            icon={Layers}
                            to="/architect"
                            delay={0.2}
                            className="md:col-span-2 group"
                        />

                        <GlassCard
                            title="Live Telemetry"
                            description="Watch the training loss converge in real-time."
                            icon={Activity}
                            to="/training"
                            delay={0.3}
                            className="md:col-span-2"
                        />

                        <GlassCard
                            title="High Performance Workers"
                            description="Under the hood, dedicated Celery ML workers process epochs asynchronously. Your UI stays silky smooth at 60fps, no matter how deep the learning gets."
                            icon={Cpu}
                            to="/"
                            delay={0.4}
                            className="md:col-span-4"
                        />
                    </div>
                </section>

                {/* Final CTA Section */}
                <section className="relative z-10 max-w-5xl mx-auto px-6 py-32 text-center">
                    <FadeIn direction="up">
                        <div className="rounded-[3rem] border border-white/10 bg-gradient-to-b from-white/5 to-transparent p-12 md:p-24 backdrop-blur-xl relative overflow-hidden">
                            <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full h-[1px] bg-gradient-to-r from-transparent via-accent-3 to-transparent" />
                            <h2 className="text-4xl md:text-5xl font-bold tracking-tight text-white mb-6">Ready to train?</h2>
                            <p className="text-lg text-text-secondary font-light mb-10 max-w-xl mx-auto">
                                Join to create, manage, and scale your machine learning experiments seamlessly from your browser.
                            </p>
                            <Link
                                to="/signup"
                                className="group inline-flex h-14 items-center justify-center gap-2 rounded-full bg-gradient-to-r from-accent-1 to-accent-3 px-8 font-semibold text-white transition-all hover:scale-[1.02] shadow-[0_0_40px_rgba(139,92,246,0.4)] hover:shadow-[0_0_60px_rgba(139,92,246,0.6)]"
                            >
                                Create Free Account &mdash; The Future awaits
                            </Link>
                        </div>
                    </FadeIn>
                </section>

            </main>
        </div>
    );
};

export default Home;
