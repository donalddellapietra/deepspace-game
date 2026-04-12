import { useEffect, useState } from "react";
import { useToast, clearToast } from "../hooks/useGameState";
import "./Toast.css";

interface ToastEntry {
  text: string;
  id: number;
  fading: boolean;
}

export function ToastContainer() {
  const [toasts, setToasts] = useState<ToastEntry[]>([]);
  const incoming = useToast();

  useEffect(() => {
    if (!incoming) return;

    const entry: ToastEntry = { ...incoming, fading: false };
    setToasts((prev) => [...prev, entry]);
    clearToast();

    // Start fade after 1.1s
    const fadeTimer = setTimeout(() => {
      setToasts((prev) =>
        prev.map((t) => (t.id === entry.id ? { ...t, fading: true } : t))
      );
    }, 1100);

    // Remove after 1.5s
    const removeTimer = setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== entry.id));
    }, 1500);

    return () => {
      clearTimeout(fadeTimer);
      clearTimeout(removeTimer);
    };
  }, [incoming]);

  if (toasts.length === 0) return null;

  return (
    <div className="toast-container">
      {toasts.map((t) => (
        <div key={t.id} className={`toast ${t.fading ? "fading" : ""}`}>
          {t.text}
        </div>
      ))}
    </div>
  );
}
