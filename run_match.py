"""Background match runner: runs N hands and saves results."""
import match
import run

if __name__ == '__main__':
    run.run_api_match = lambda *args, **kwargs: match.run_api_match(*args, **kwargs, num_hands=8)
    run.main()

