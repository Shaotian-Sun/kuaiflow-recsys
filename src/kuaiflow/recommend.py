"""Generate recommendations using the saved validation-selected pipeline."""
import argparse
import torch
from kuaiflow.pipeline import RecommendationPipeline


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline',default='artifacts/pipeline_improvement/selected_pipeline.json')
    parser.add_argument('--users',type=int,nargs='+',required=True)
    parser.add_argument('--k',type=int,default=20)
    parser.add_argument('--output',help='Optional CSV path; otherwise print CSV')
    args=parser.parse_args()
    torch.set_num_threads(1)
    result=RecommendationPipeline.load(args.pipeline).recommend(args.users,args.k)
    result=result[['user_id','video_id','final_rank','final_score']]
    if args.output:
        result.to_csv(args.output,index=False)
    else:
        print(result.to_csv(index=False),end='')


if __name__=='__main__':
    main()
