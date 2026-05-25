import './globals.css';

export const metadata = {
  title: 'Neon Calculator',
  description: 'A beautiful calculator built with HTML, CSS, React and Next.js.'
};

export default function RootLayout({ children }) {
  return (
    <html lang="de">
      <body>{children}</body>
    </html>
  );
}
