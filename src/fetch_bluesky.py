# import re #load a "regular expression" library for helping to parse text
# from atproto import IdResolver # Load the atproto IdResolver library to get offical ATProto user IDs

import re
import json
from datetime import datetime, timezone
from dateutil import parser
from atproto import Client
import networkx as nx

import matplotlib as mpl
import matplotlib.pyplot as plt


# Bluesky credentials
BLUESKY_USERNAME = "theultimateminer.bsky.social"  # Replace with your Bluesky handle/email
BLUESKY_PASSWORD = "the_ultimate_m1n3r!"  # Replace with your app password
BLUESKY_FEED_PASSWORD = "fsbt-rffp-gt3o-osfe"

# API endpoints
LOGIN_URL = "https://bsky.social/xrpc/com.atproto.server.createSession"
SEARCH_URL = "https://bsky.social/xrpc/app.bsky.feed.searchPosts"
FOLLOWING_URL = "https://bsky.social/xrpc/app.bsky.graph.getFollows"
FOLLOWERS_URL = "https://bsky.social/xrpc/app.bsky.graph.getFollowers"


# Define thresholds
CREATION_CUTOFF = datetime(2024, 7, 1, tzinfo=timezone.utc)

# Feed link (generate hashtag feed from track.goodfeeds.co)
#feedUrl = "https://bsky.app/profile/did:plc:lyrmsmhhg7vzz4ghj44y5xzq/feed/70f49077560c"
feedUrl = "https://bsky.app/profile/did:plc:lyrmsmhhg7vzz4ghj44y5xzq/feed/5aa097406d64"



# Define the date range
DATE_START = "2025-02-27"
DATE_END = "2025-02-28"

def parse_date(date_str):
    return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)

def parse_datetime(date_str):
    """Parses timestamps with or without microseconds."""
    # try:
    #     return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ")  # With microseconds
    # except ValueError:
    #     return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")  # Without microseconds
    return parser.isoparse(date_str)


# function to convert a feed from a weblink url to the special atproto "at" URI
def getATFeedLinkFromURL(url):
    
    # Get the user did and feed id from the weblink url
    match = re.search(r'https://bsky.app/profile/([^/]+)/feed/([^/]+)', url)

    if not match:
        raise ValueError("Invalid Bluesky feed URL format.")
    did, feed_id = match.groups()

    # NB: assumes that the feedURL is already in "did:xxx" format.

    # Construct the at:// URI
    post_uri = f"at://{did}/app.bsky.feed.generator/{feed_id}"

    return post_uri

# function to convert a post's special atproto "at" URI to a weblink url
def getWebLinkFromPost(post):
    # Get the user id and post id from the weblink url
    match = re.search(r'at://([^/]+)/app.bsky.feed.post/([^/]+)', post.uri)
    if not match:
        raise ValueError("Invalid Bluesky atproto post URL format.")
    user_id, post_id = match.groups()

    post_uri = f"https://bsky.app/profile/{user_id}/post/{post_id}"
    return post_uri


def get_user_follow_data(client, handle):
    """Fetches followers and following lists for a given user handle."""
    follows = client.app.bsky.graph.get_follows({'actor': handle}).follows
    followers = client.app.bsky.graph.get_followers({'actor': handle}).followers
    return {user.handle for user in follows}, {user.handle for user in followers}

def get_user_metadata(client, handle):
    """Fetches user metadata like account creation date and follower-following ratio."""
    profile = client.app.bsky.actor.get_profile({'actor': handle})
    created_at = parse_datetime(profile.created_at)
    return {
        'created_at': "old" if created_at < CREATION_CUTOFF else "new",
        'follower_following_ratio': "high" if profile.followers_count / max(profile.follows_count, 1) > 1 else "low"
    }

from atproto import Client
client = Client(base_url="https://bsky.social")
client.login(BLUESKY_USERNAME,BLUESKY_PASSWORD)

# Generate atproto link for feed
atFeedLink = getATFeedLinkFromURL(feedUrl)

# Test: get first post

# feed = client.app.bsky.feed.get_feed({'feed': atFeedLink}).feed
# recent_post = feed[0].post
# # post text
# print("The post text is: " + recent_post.record.text)
# # author handle
# print("The author handle is: " + str(recent_post.author.handle))
# # post id
# print("The post content id is: " + str(recent_post.cid))
# # num likes
# print("The number of likes is: " + str(recent_post.like_count))
# # num reposts
# print("The number of reposts is: " + str(recent_post.repost_count))


# Get feed posts
feed = client.app.bsky.feed.get_feed({'feed': atFeedLink}).feed

# Filter posts within the date range
G = nx.Graph()
users = {}
uid=0

# for post in feed:
#     #post_date = datetime.strptime(post.post.record.created_at, "%Y-%m-%dT%H:%M:%S.%fZ")
#     post_date = parse_datetime(post.post.record.created_at)

#     if parse_date(DATE_START) <= post_date <= parse_date(DATE_END):
#         author = post.post.author.handle
#         if author not in users:
#             follows, followers = get_user_follow_data(client, author)

#             metadata = get_user_metadata(client, author)

#             users[author] = {
#                 'uid':uid,
#                 'follows': follows,
#                 'followers': followers,
#                 'created_at': metadata['created_at'],
#                 'ratio': metadata['follower_following_ratio']
#             }
#             uid+=1

#             # Store metadata for all followers and following
#             for user in follows.union(followers):
#                 if user not in users:
#                     metadata = get_user_metadata(client, user)
#                     users[user] = {
#                         'uid':uid,
#                         'follows': set(),
#                         'followers': set(),
#                         'created_at': metadata['created_at'],
#                         'ratio': metadata['follower_following_ratio']
#                     }
#                 uid+=1

#         # @DEBUG
#         # print(users[author])


# # @TODO: After test: fetch all data, save as pickle and create network from there

# # Create an edge list for connections
# for user, data in users.items():
#     # add edges for users one follows:
#     for followed in data['follows']:
#         G.add_edge(users[user]['uid'], users[followed]['uid'], weight=1.0)
#     # add edges for users one is followed by:
#     for follower in data['followers']:
#         G.add_edge(users[user]['uid'], users[follower]['uid'], weight=1.0)


uid = 0

for post in feed:
    post_date = parse_datetime(post.post.record.created_at)
    if parse_date(DATE_START) <= post_date <= parse_date(DATE_END):
        author = post.post.author.handle
        if author not in users:
            metadata = get_user_metadata(client, author)
            users[author] = {
                'uid': uid,
                'created_at': metadata['created_at'],
                'ratio': metadata['follower_following_ratio']
            }
            uid += 1
        
        # Process replies and reposts safely
        if hasattr(post.post.record, 'reply') and post.post.record.reply:
            if hasattr(post.post.record.reply, 'parent') and hasattr(post.post.record.reply.parent, 'author'):
                replied_to = post.post.record.reply.parent.author.handle
                if replied_to not in users:
                    metadata = get_user_metadata(client, replied_to)
                    users[replied_to] = {
                        'uid': uid,
                        'created_at': metadata['created_at'],
                        'ratio': metadata['follower_following_ratio']
                    }
                    uid += 1
                G.add_edge(users[author]['uid'], users[replied_to]['uid'])
        
        if hasattr(post.post.record, 'repost') and post.post.record.repost:
            if hasattr(post.post.record.repost, 'author'):
                reposted = post.post.record.repost.author.handle
                if reposted not in users:
                    metadata = get_user_metadata(client, reposted)
                    users[reposted] = {
                        'uid': uid,
                        'created_at': metadata['created_at'],
                        'ratio': metadata['follower_following_ratio']
                    }
                    uid += 1
                G.add_edge(users[author]['uid'], users[reposted]['uid'])


# Store node attributes (creation date and follower/following ratio)
nx.set_node_attributes(G, {data['uid']: data['created_at'] for user, data in users.items()}, name='created_at')
nx.set_node_attributes(G, {data['uid']: data['ratio'] for user, data in users.items()}, name='follower_following_ratio')

# Test: draw graph
nx.draw(G)
plt.show()

# Save the edge list
nx.write_edgelist(G, "bluesky_edgelist.txt")
print("Edge list and node attributes successfully generated.")


