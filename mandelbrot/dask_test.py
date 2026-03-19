from dask.distributed import Client
import time

def main():
    client = Client()
    print(client)
    print("dashboard:", client.dashboard_link)

    future = client.submit(lambda x: x + 1, 10)
    print(future.result())

    print("Keeping dashboard alive for 60 seconds...")
    time.sleep(60)

if __name__ == "__main__":
    main()