from . import data_page, filter_page, plot_page, baseline_page, exports_page

PAGES = {
    "Data": data_page.render,
    "Filter": filter_page.render,
    "Plot & Compare": plot_page.render,
    "Baseline Shift": baseline_page.render,
    "Exports": exports_page.render,
}
