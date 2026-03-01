import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Home from './pages/Home';
import Dataset from './pages/Dataset';
import Architect from './pages/Architect';
import Training from './pages/Training';

const App: React.FC = () => {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Home />} />
        <Route path="dataset" element={<Dataset />} />
        <Route path="architect" element={<Architect />} />
        <Route path="training" element={<Training />} />
      </Route>
    </Routes>
  );
};

export default App;
