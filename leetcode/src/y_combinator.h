#pragma once

template <class F> struct y_combinator {
	F f; // the lambda will be stored here

	// a forwarding operator():
	template <class... Args>
	decltype(auto) operator()(Args &&... args) const
	{
		// we pass ourselves to f, then the arguments.
		// the lambda should take the first argument as `auto&& recurse` or similar.
		return f(*this, std::forward<Args>(args)...);
	}
};

// helper function that deduces the type of the lambda:
template <class F> y_combinator<std::decay_t<F> > make_y_combinator(F &&f)
{
	return { std::forward<F>(f) };
}

// (Be aware that in C++17 we can do better than a `make_` function)
#if !(_HAS_CXX20)
template <class F> y_combinator(F) -> y_combinator<F>;
#endif