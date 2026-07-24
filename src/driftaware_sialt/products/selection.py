"""Selection of date-matched auxiliary products."""

import datetime


def select_product(products, t0, t1, max_time_difference_days=7):
    """Return the priority exact match or closest product within the limit."""
    for product in products:
        product.target_files = None
    exact = []
    nearby = []
    limit = datetime.timedelta(days=max_time_difference_days)

    for priority, product in enumerate(products):
        for file, date in zip(product.file_list, product.file_dates):
            candidate = (priority, date > t0, date, file, product)
            if t0 <= date < t1:
                exact.append(candidate)
            elif abs(date - t0) <= limit:
                nearby.append((abs(date - t0),) + candidate)

    if exact:
        _, _, date, file, product = min(exact)
        fallback = False
    elif nearby:
        _, _, _, date, file, product = min(nearby)
        fallback = True
    else:
        return None

    product.target_files = file
    product.target_file_date = date
    product.target_time_offset_days = (
        date - t0).total_seconds() / 86400.0
    product.used_temporal_fallback = fallback
    return product
