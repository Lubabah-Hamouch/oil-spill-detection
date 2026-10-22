const API_URL = (import.meta.env.VITE_API_URL ?? "http://localhost:8000").replace(/\/+$/, "");

export type ConfidenceLabel = "NO_SPILL" | "LOW_CONFIDENCE_SPILL" | "HIGH_CONFIDENCE_SPILL";

/** Response body of POST /predict. */
export interface PredictResponse {
  probability: number;
  binary_label: 0 | 1;
  confidence_label: ConfidenceLabel;
  spill_detected: boolean;
  segmentation_status: string;
  overlay_png_base64: string | null;
  area_percentage: number | null;
}

/** PredictResponse with the overlay decoded into something an <img> can display. */
export interface PredictResult extends Omit<PredictResponse, "overlay_png_base64"> {
  overlayUrl: string | null;
}

export class ApiError extends Error {
  readonly status?: number;
  constructor(message: string, status?: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function readErrorDetail(res: Response): Promise<string | null> {
  try {
    const body = await res.json();
    return typeof body?.detail === "string" ? body.detail : null;
  } catch {
    return null;
  }
}

/** Upload an image to the backend and return the analysis. Throws ApiError with a user-presentable message. */
export async function predict(file: File, signal?: AbortSignal): Promise<PredictResult> {
  const form = new FormData();
  form.append("file", file);

  let res: Response;
  try {
    res = await fetch(`${API_URL}/predict`, { method: "POST", body: form, signal });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new ApiError("Couldn't reach the analysis service. Check that the backend is running and try again.");
  }

  if (!res.ok) {
    const detail = await readErrorDetail(res);
    if (detail) throw new ApiError(detail, res.status);
    throw new ApiError(
      res.status >= 500
        ? "The analysis service ran into a problem. Please try again."
        : "The image couldn't be analyzed. Please try a different file.",
      res.status,
    );
  }

  let data: PredictResponse;
  try {
    data = (await res.json()) as PredictResponse;
  } catch {
    throw new ApiError("Received an unexpected response from the analysis service.", res.status);
  }

  const { overlay_png_base64, ...rest } = data;
  return {
    ...rest,
    overlayUrl: overlay_png_base64 ? `data:image/png;base64,${overlay_png_base64}` : null,
  };
}
