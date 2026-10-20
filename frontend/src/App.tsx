/**
 * App root: react-query provider + router.
 *
 * Query defaults: 1 retry (the backend may briefly hiccup during model
 * load), 10s stale time (health/model card are cheap but not free).
 * `/` redirects to the analyze wizard; unmatched paths render a plain
 * 404 card with a way back.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  BrowserRouter,
  Link,
  Navigate,
  Route,
  Routes,
} from "react-router-dom";
import { Layout } from "./components/Layout";
import { AboutPage } from "./pages/AboutPage";
import { AnalyzePage } from "./pages/AnalyzePage";
import { PatientPage } from "./pages/PatientPage";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      staleTime: 10_000,
    },
  },
});

function NotFoundPage() {
  return (
    <section className="rounded-card border border-hairline bg-white p-8 text-center shadow-card">
      <h1 className="text-[18px] font-semibold text-ink">Page not found</h1>
      <p className="mt-2 text-[13px] text-ink2">
        The page you requested does not exist.
      </p>
      <Link
        to="/analyze"
        className="mt-4 inline-block rounded-control bg-primary px-3 py-1.5 text-[13px] font-medium text-white hover:bg-primary-dark"
      >
        Back to Analyze
      </Link>
    </section>
  );
}

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route path="/" element={<Navigate to="/analyze" replace />} />
            <Route path="/analyze" element={<AnalyzePage />} />
            <Route path="/patient" element={<PatientPage />} />
            <Route path="/about" element={<AboutPage />} />
            <Route path="*" element={<NotFoundPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
