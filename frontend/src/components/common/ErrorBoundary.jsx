import React from 'react';

/**
 * Error Boundary Component
 * Catches JavaScript errors in child component tree and displays a fallback UI
 */
class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false, error: null, errorInfo: null };
    }

    static getDerivedStateFromError(error) {
        // Update state so the next render will show the fallback UI.
        return { hasError: true, error };
    }

    componentDidCatch(error, errorInfo) {
        // You can also log the error to an error reporting service
        console.error("ErrorBoundary caught an error", error, errorInfo);
        this.setState({ errorInfo });
    }

    handleReset = () => {
        this.setState({ hasError: false, error: null, errorInfo: null });
        if (this.props.onReset) {
            this.props.onReset();
        }
    };

    render() {
        if (this.state.hasError) {
            if (this.props.fallback) {
                return this.props.fallback;
            }

            return (
                <div className="error-boundary-container">
                    <div className="error-content">
                        <div className="error-icon">⚠️</div>
                        <h2>Something went wrong</h2>
                        <p>
                            {this.props.errorMessage || "We encountered an unexpected error while rendering this view."}
                        </p>
                        {this.state.error && (
                            <details className="error-details">
                                <summary>Error Details</summary>
                                <pre>{this.state.error.toString()}</pre>
                            </details>
                        )}
                        <button className="btn btn-primary" onClick={this.handleReset}>
                            Try Again
                        </button>
                    </div>
                    <style>{`
            .error-boundary-container {
              display: flex;
              align-items: center;
              justify-content: center;
              height: 100%;
              width: 100%;
              min-height: 400px;
              background: var(--color-bg-primary, #050508);
              padding: 20px;
            }
            .error-content {
              display: flex;
              flex-direction: column;
              align-items: center;
              max-width: 400px;
              text-align: center;
              gap: 16px;
              background: var(--color-bg-secondary, #0a0a0f);
              padding: 32px;
              border-radius: 16px;
              border: 1px solid var(--color-border, #333);
            }
            .error-icon {
              font-size: 48px;
            }
            .error-content h2 {
              font-size: 24px;
              font-weight: 600;
              margin: 0;
              color: var(--color-text-primary, #fff);
            }
            .error-content p {
              color: var(--color-text-secondary, #aaa);
              margin: 0;
            }
            .error-details {
                text-align: left;
                width: 100%;
                background: #000;
                padding: 10px;
                border-radius: 8px;
                font-family: monospace;
                font-size: 12px;
                color: #ff6666;
                overflow: auto;
            }
            .btn {
              padding: 10px 20px;
              background: var(--color-accent, #00d4aa);
              border: none;
              border-radius: 8px;
              color: black;
              font-weight: 600;
              cursor: pointer;
              margin-top: 8px;
            }
            .btn:hover {
                opacity: 0.9;
            }
          `}</style>
                </div>
            );
        }

        return this.props.children;
    }
}

export default ErrorBoundary;
