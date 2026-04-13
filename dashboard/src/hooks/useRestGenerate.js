import { useCallback, useState } from "react";

const apiHttp =
  (import.meta.env.VITE_API_HTTP_TARGET || "http://127.0.0.1:8010").replace(/\/$/, "");

export function useRestGenerate() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const generate = useCallback(async (model, prompt, maxNewTokens, detectSuppression) => {
    setLoading(true);
    setError(null);
    try {
      const r = await fetch(`${apiHttp}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model,
          prompt,
          max_new_tokens: maxNewTokens,
          detect_suppression: detectSuppression,
        }),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return await r.json();
    } catch (e) {
      setError(String(e));
      return null;
    } finally {
      setLoading(false);
    }
  }, []);

  return { generate, loading, error };
}
