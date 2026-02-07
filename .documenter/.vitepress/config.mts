import { defineConfig } from 'vitepress'
import { tabsMarkdownPlugin } from 'vitepress-plugin-tabs'
import { mathjaxPlugin } from './mathjax-plugin'
import footnote from "markdown-it-footnote";
import path from 'path'

const mathjax = mathjaxPlugin()

function getBaseRepository(base: string): string {
  if (!base || base === '/') return '/';
  const parts = base.split('/').filter(Boolean);
  return parts.length > 0 ? `/${parts[0]}/` : '/';
}

const baseTemp = {
  base: '/QuantNova.jl/dev/',// TODO: replace this in makedocs!
}

const navTemp = {
  nav: [
{ text: 'Home', link: '/index' },
{ text: 'Getting Started', collapsed: false, items: [
{ text: 'Installation', link: '/getting-started/installation' },
{ text: 'Quick Start', link: '/getting-started/quickstart' }]
 },
{ text: 'Examples', collapsed: false, items: [
{ text: 'Pricing and Calibration Case Study', link: '/examples/pricing-calibration-demo' },
{ text: 'Backtesting Demo', link: '/examples/backtesting-demo' },
{ text: 'Portfolio Optimization Demo', link: '/examples/optimization-demo' },
{ text: 'Option Pricing Walkthrough', link: '/examples/option-pricing' },
{ text: 'Portfolio Risk Management', link: '/examples/portfolio-risk' },
{ text: 'Monte Carlo Exotic Options', link: '/examples/monte-carlo-exotic' },
{ text: 'Yield Curve Construction', link: '/examples/yield-curve' }]
 },
{ text: 'Manual', collapsed: false, items: [
{ text: 'AD Backends', link: '/manual/backends' },
{ text: 'Benchmark Methodology', link: '/manual/benchmarks' },
{ text: 'Validation', link: '/manual/validation' },
{ text: 'Monte Carlo Simulation', link: '/manual/montecarlo' },
{ text: 'Portfolio Optimization', link: '/manual/optimization' },
{ text: 'Backtesting', link: '/manual/backtesting' },
{ text: 'Interest Rates', link: '/manual/interest-rates' },
{ text: 'Scenario Analysis', link: '/manual/scenario-analysis' },
{ text: 'Simulation Engine', link: '/manual/simulation' }]
 },
{ text: 'API Reference', link: '/api' }
]
,
}

const nav = [
  ...navTemp.nav,
  {
    component: 'VersionPicker'
  }
]

// https://vitepress.dev/reference/site-config
export default defineConfig({
  base: '/QuantNova.jl/dev/',// TODO: replace this in makedocs!
  title: 'QuantNova.jl',
  description: 'Documentation for QuantNova.jl',
  lastUpdated: true,
  cleanUrls: true,
  ignoreDeadLinks: ['./@ref'],
  outDir: '../1', // This is required for MarkdownVitepress to work correctly...
  head: [
    
    ['script', {src: `${getBaseRepository(baseTemp.base)}versions.js`}],
    // ['script', {src: '/versions.js'], for custom domains, I guess if deploy_url is available.
    ['script', {src: `${baseTemp.base}siteinfo.js`}]
  ],
  
  markdown: {
    config(md) {
      md.use(tabsMarkdownPlugin);
      md.use(footnote);
      mathjax.markdownConfig(md);
    },
    theme: {
      light: "github-light",
      dark: "github-dark"
    },
  },
  vite: {
    plugins: [
      mathjax.vitePlugin,
    ],
    define: {
      __DEPLOY_ABSPATH__: JSON.stringify('/QuantNova.jl'),
    },
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '../components')
      }
    },
    optimizeDeps: {
      exclude: [ 
        '@nolebase/vitepress-plugin-enhanced-readabilities/client',
        'vitepress',
        '@nolebase/ui',
      ], 
    }, 
    ssr: { 
      noExternal: [ 
        // If there are other packages that need to be processed by Vite, you can add them here.
        '@nolebase/vitepress-plugin-enhanced-readabilities',
        '@nolebase/ui',
      ], 
    },
  },
  themeConfig: {
    outline: 'deep',
    
    search: {
      provider: 'local',
      options: {
        detailedView: true
      }
    },
    nav,
    sidebar: [
{ text: 'Home', link: '/index' },
{ text: 'Getting Started', collapsed: false, items: [
{ text: 'Installation', link: '/getting-started/installation' },
{ text: 'Quick Start', link: '/getting-started/quickstart' }]
 },
{ text: 'Examples', collapsed: false, items: [
{ text: 'Pricing and Calibration Case Study', link: '/examples/pricing-calibration-demo' },
{ text: 'Backtesting Demo', link: '/examples/backtesting-demo' },
{ text: 'Portfolio Optimization Demo', link: '/examples/optimization-demo' },
{ text: 'Option Pricing Walkthrough', link: '/examples/option-pricing' },
{ text: 'Portfolio Risk Management', link: '/examples/portfolio-risk' },
{ text: 'Monte Carlo Exotic Options', link: '/examples/monte-carlo-exotic' },
{ text: 'Yield Curve Construction', link: '/examples/yield-curve' }]
 },
{ text: 'Manual', collapsed: false, items: [
{ text: 'AD Backends', link: '/manual/backends' },
{ text: 'Benchmark Methodology', link: '/manual/benchmarks' },
{ text: 'Validation', link: '/manual/validation' },
{ text: 'Monte Carlo Simulation', link: '/manual/montecarlo' },
{ text: 'Portfolio Optimization', link: '/manual/optimization' },
{ text: 'Backtesting', link: '/manual/backtesting' },
{ text: 'Interest Rates', link: '/manual/interest-rates' },
{ text: 'Scenario Analysis', link: '/manual/scenario-analysis' },
{ text: 'Simulation Engine', link: '/manual/simulation' }]
 },
{ text: 'API Reference', link: '/api' }
]
,
    editLink: { pattern: "https://github.com/KookiesNKareem/QuantNova.jl/edit/main/docs/src/:path" },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/KookiesNKareem/QuantNova.jl' }
    ],
    footer: {
      message: 'Made with <a href="https://luxdl.github.io/DocumenterVitepress.jl/dev/" target="_blank"><strong>DocumenterVitepress.jl</strong></a><br>',
      copyright: `© Copyright ${new Date().getUTCFullYear()}.`
    }
  }
})
