import { useCallback, useState } from "react";
import { apiBase } from "../apiBase.js";

export function useRestGenerate() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const generate = useCallback(async (model, prompt, maxNewTokens, detectSuppression, guardMode = "observe", interventionMode = "none") => {
    setLoading(true);
    setError(null);
    try {
      const r = await fetch(`${apiBase()}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model,
          prompt,
          max_new_tokens: maxNewTokens,
          detect_suppression: detectSuppression,
          guard_mode: guardMode,
          intervention_mode: interventionMode,
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
