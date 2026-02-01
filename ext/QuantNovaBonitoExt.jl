module QuantNovaBonitoExt

using QuantNova
using QuantNova.Visualization: VisualizationSpec, Dashboard, Row, get_theme,
    DARK_THEME, LIGHT_THEME, render
using Bonito
using WGLMakie

"""
    serve(spec::VisualizationSpec; port=8080)

Serve a single visualization in the browser.
"""
function QuantNova.Visualization.serve(spec::VisualizationSpec; port::Int=8080)
    app = App() do session
        fig = render(spec)
        DOM.div(fig)
    end

    server = Bonito.Server(app, "0.0.0.0", port)
    println("Dashboard running at http://localhost:$port")
    server
end

"""
    serve(dashboard::Dashboard; port=8080)

Serve a multi-panel dashboard in the browser.
"""
function QuantNova.Visualization.serve(dashboard::Dashboard; port::Int=8080)
    theme = dashboard.theme == :dark ? DARK_THEME : LIGHT_THEME

    app = App() do session
        rows = map(dashboard.layout) do row
            items = map(row.items) do item
                if item isa VisualizationSpec
                    fig = render(item)
                    DOM.div(fig, style="flex: 1;")
                else
                    DOM.div(item)
                end
            end
            # Apply row weight as flex-grow
            DOM.div(items..., style="display: flex; flex-direction: row; gap: 16px; flex: $(row.weight);")
        end

        DOM.div(
            DOM.h1(dashboard.title, style="color: $(theme[:textcolor]);"),
            DOM.div(rows..., style="display: flex; flex-direction: column; gap: 16px; flex: 1;"),
            style="background: $(theme[:backgroundcolor]); padding: 24px; min-height: 100vh; display: flex; flex-direction: column;"
        )
    end

    server = Bonito.Server(app, "0.0.0.0", port)
    println("Dashboard running at http://localhost:$port")
    server
end

"""
    serve(dashboard::Dashboard, update_fn::Function; port=8080)

Serve a live-updating dashboard.
"""
function QuantNova.Visualization.serve(dashboard::Dashboard, update_fn::Function; port::Int=8080)
    theme = dashboard.theme == :dark ? DARK_THEME : LIGHT_THEME

    app = App() do session
        # Call update function with session for live data setup
        update_fn(session)

        # Build actual dashboard content
        rows = map(dashboard.layout) do row
            items = map(row.items) do item
                if item isa VisualizationSpec
                    fig = render(item)
                    DOM.div(fig, style="flex: 1;")
                else
                    DOM.div(item)
                end
            end
            DOM.div(items..., style="display: flex; flex-direction: row; gap: 16px; flex: $(row.weight);")
        end

        DOM.div(
            DOM.h1(dashboard.title, style="color: $(theme[:textcolor]);"),
            DOM.div(rows..., style="display: flex; flex-direction: column; gap: 16px; flex: 1;"),
            style="background: $(theme[:backgroundcolor]); padding: 24px; min-height: 100vh; display: flex; flex-direction: column;"
        )
    end

    server = Bonito.Server(app, "0.0.0.0", port)
    println("Live dashboard running at http://localhost:$port")
    server
end

end # module
