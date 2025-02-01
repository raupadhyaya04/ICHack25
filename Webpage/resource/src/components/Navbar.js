// src/components/Navbar.js
import React from 'react';
import { Link } from 'react-router-dom';

export default function Navbar() {
  return (
    <nav>
      <ul>
        <li><Link to="/">Home</Link></li>
        <li><Link to="/distributors">Distributors</Link></li>
        <li><Link to="/suppliers">Suppliers</Link></li>
      </ul>
    </nav>
  );
}