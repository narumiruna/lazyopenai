from dotenv import load_dotenv

from lazyopenai.chat import create


def main() -> None:
    load_dotenv()

    response = create(messages="Hi.")
    print(response)
    response = create(messages=["What date is today?"])
    print(response)
    response = create(messages=[{"role": "user", "content": "What is your name?"}])
    print(response)


if __name__ == "__main__":
    main()
