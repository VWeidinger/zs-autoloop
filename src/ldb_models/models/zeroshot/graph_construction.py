from ldb_models.models.zeroshot.encoding import encode, get_or_add_col_id, parse_predicates


def plan_to_graph(
    node,
    run_file_id: int,
    plan_depths,
    plan_features,
    subplan_labels,
    subplan_card_labels,
    subplan_est_card_labels,
    plan_to_plan_edges,
    db_statistics,
    feature_statistics,
    filter_to_plan_edges,
    predicate_col_features,
    output_column_to_plan_edges,
    output_column_features,
    column_to_output_column_edges,
    column_to_filter_edges,
    column_features,
    table_features,
    table_to_plan_edges,
    output_column_idx,
    column_idx,
    table_idx,
    plan_featurization,
    predicate_depths,
    intra_predicate_edges,
    logical_preds,
    parent_node_id=None,
    depth=0,
    skip_top_aggregation: bool = False,
    filter_inline_col_feats: bool = True,
    annotate_base_table_card_to_scan_op: bool = False,
    subplan_label_op_time: bool = False,
    sumo_loop: bool = False,
    sumo_loop_multiply_est_error: bool = False,
    sumo_loop_loss_on_scaled: bool = False,
    loop_multiplier_map: tuple = None,
):
    plan_params = node["plan_parameters"]

    # Initialize loop_multiplier_map on first call (tuple of two lists: node_ids, sibling_ids)
    if loop_multiplier_map is None:
        loop_multiplier_map = ([], [])

    if "Aggregate" in plan_params["op_name"] and skip_top_aggregation:
        assert len(node["children"]) == 1, (
            f"Aggregation has more than one child: {node}"
        )
        plan_to_graph(
            node["children"][0],
            run_file_id,
            plan_depths,
            plan_features,
            subplan_labels,
            subplan_card_labels,
            subplan_est_card_labels,
            plan_to_plan_edges,
            db_statistics,
            feature_statistics,
            filter_to_plan_edges,
            predicate_col_features,
            output_column_to_plan_edges,
            output_column_features,
            column_to_output_column_edges,
            column_to_filter_edges,
            column_features,
            table_features,
            table_to_plan_edges,
            output_column_idx,
            column_idx,
            table_idx,
            plan_featurization,
            predicate_depths,
            intra_predicate_edges,
            logical_preds,
            parent_node_id=parent_node_id,
            depth=depth,
            skip_top_aggregation=False,
            filter_inline_col_feats=filter_inline_col_feats,
            annotate_base_table_card_to_scan_op=annotate_base_table_card_to_scan_op,
            subplan_label_op_time=subplan_label_op_time,
            sumo_loop=sumo_loop,
            sumo_loop_multiply_est_error=sumo_loop_multiply_est_error,
            sumo_loop_loss_on_scaled=sumo_loop_loss_on_scaled,
            loop_multiplier_map=loop_multiplier_map,
        )
        return

    plan_node_id = len(plan_depths)

    subplan_card_labels.append(plan_params.get("act_card", 1.0))
    subplan_est_card_labels.append(plan_params.get("est_card", 1.0))

    # Extract the ground-truth runtime for this subplan (plan node)
    if subplan_label_op_time:
        if "act_time" in plan_params:
            act_time = plan_params["act_time"]
            children_act_times = [
                child["plan_parameters"].get("act_time", 0)
                for child in node["children"]
            ]
            sum_child_time = sum(children_act_times) if children_act_times else 0

            op_time = act_time - sum_child_time
            # Clamp at 0, as operation time cannot be negative (due to measurement noise)
            op_time = max(op_time, 1e-6)

            # SUMO-LOOP default mode: train on per-iteration operator time.
            # PG-offset and scaled-loss modes keep loop nodes in full-runtime space.
            if (
                sumo_loop
                and not sumo_loop_multiply_est_error
                and not sumo_loop_loss_on_scaled
                and plan_params.get("act_loops", 1) > 1
            ):
                act_loops = plan_params["act_loops"]
                op_time_per_loop = op_time / act_loops
                subplan_labels.append(max(op_time_per_loop, 1e-6))
            else:
                subplan_labels.append(op_time)
        else:
            # Cannot compute op_time without act_time
            print("No act_time in plan parameters for op_time calculation")
            subplan_labels.append(1e-6)
    else:
        if "act_time" in plan_params:
            if len(node["children"]) > 0:
                children_act_times = [
                    child["plan_parameters"]["act_time"] for child in node["children"]
                ]
                sum_child_time = sum(children_act_times) if children_act_times else 0
                if plan_params["act_time"] < sum_child_time:
                    subplan_labels.append(sum_child_time)
                else:
                    subplan_labels.append(plan_params["act_time"])
            else:
                subplan_labels.append(plan_params["act_time"])
        else:
            print("No act_time in plan parameters")
            if len(node["children"]) > 0:
                children_act_times = [
                    child["plan_parameters"]["act_time"] for child in node["children"]
                ]
                print(f"Using children act_time: {children_act_times}")
                sum_child_time = sum(children_act_times) if children_act_times else 0
                subplan_labels.append(sum_child_time)
            else:
                subplan_labels.append(1e-6)

    is_scan_op = "table_id" in plan_params

    # add plan features
    curr_plan_features = []
    for feature_name in plan_featurization.PLAN_FEATURES:
        if (
            feature_name in ["est_children_card", "act_children_card"]
            and is_scan_op
            and annotate_base_table_card_to_scan_op
        ):
            # extract base table card
            base_table_card = db_statistics[run_file_id]["table_stats"][
                plan_params["table_id"]
            ]["reltuples"]

            # annotate the base table card to the scan op
            curr_plan_features.append(
                encode(
                    "base_table_card",
                    {feature_name: base_table_card},
                    feature_statistics,
                )
            )
        else:
            curr_plan_features.append(
                encode(feature_name, plan_params, feature_statistics)
            )

    plan_features.append(curr_plan_features)
    plan_depths.append(depth)

    # encode output columns which can in turn have several columns as a product in the aggregation
    output_columns = plan_params.get("output_columns")
    if output_columns is not None:
        for output_column in output_columns:
            output_column_id_list = [col["col_id"] for col in output_column["columns"]]
            output_column_node_id = output_column_idx.get(
                (output_column["aggregation"], str(output_column_id_list), run_file_id)
            )

            # if not, create
            if output_column_node_id is None:
                curr_output_column_features = [
                    encode(column, output_column, feature_statistics)
                    for column in plan_featurization.OUTPUT_COLUMN_FEATURES
                ]

                output_column_node_id = len(output_column_features)
                output_column_features.append(curr_output_column_features)
                output_column_idx[
                    (
                        output_column["aggregation"],
                        str(output_column_id_list),
                        run_file_id,
                    )
                ] = output_column_node_id

                # featurize product of columns if there are any
                run_file_column_features = db_statistics[run_file_id]["column_stats"]
                assert run_file_column_features is not None
                for column in output_column["columns"]:
                    # lookup column node id or create node if not yet existing
                    column_node_id = get_or_add_col_id(
                        column["col_id"],
                        column_idx,
                        run_file_id,
                        run_file_column_features=run_file_column_features,
                        feature_statistics=feature_statistics,
                        plan_featurization=plan_featurization,
                        column_features=column_features,
                    )
                    column_to_output_column_edges.append(
                        (column_node_id, output_column_node_id)
                    )

            # in any case add the corresponding edge
            output_column_to_plan_edges.append((output_column_node_id, plan_node_id))

    # filter_columns (we do not reference the filter columns to columns since we anyway have to create a node per filter
    #  node)
    filter = plan_params.get("filter")
    if filter is not None:
        assert "is_index_cond" in filter, (
            f"No is_index_cond in filter: {filter} / {node}"
        )
        db_column_features = db_statistics[run_file_id]["column_stats"]
        try:
            parse_predicates(
                db_column_features,
                feature_statistics,
                filter,
                filter_to_plan_edges,
                column_to_filter_edges=column_to_filter_edges,
                plan_featurization=plan_featurization,
                predicate_col_features=predicate_col_features,
                predicate_depths=predicate_depths,
                intra_predicate_edges=intra_predicate_edges,
                logical_preds=logical_preds,
                column_features=column_features,
                column_idx=column_idx,
                run_file_id=run_file_id,
                plan_node_id=plan_node_id,
                inline_col_feats=filter_inline_col_feats,
            )
        except Exception as e:
            print(f"Error in filter: {filter}", flush=True)
            raise e

    # featurize join conditions
    join = plan_params.get("join")

    # check if JOIN is a condition of the form: t1.c1 = t1.c2 (i.e. it is a filter rather than a join)
    # in this case skip this join
    if is_scan_op and join is not None:
        if join["table_name1"] == join["table_name2"]:
            # this is a filter rather than a join
            # currently we cannot featurize this
            # skip this
            join = None

    if join is not None:
        assert "is_index_cond" in join, f"No is_index_cond in join: {join} / {node}"
        run_file_column_features = db_statistics[run_file_id]["column_stats"]
        try:
            parse_predicates(
                run_file_column_features,
                feature_statistics,
                join,
                filter_to_plan_edges,
                column_to_filter_edges=column_to_filter_edges,
                plan_featurization=plan_featurization,
                predicate_col_features=predicate_col_features,
                predicate_depths=predicate_depths,
                intra_predicate_edges=intra_predicate_edges,
                logical_preds=logical_preds,
                column_features=column_features,
                column_idx=column_idx,
                run_file_id=run_file_id,
                plan_node_id=plan_node_id,
                is_join=True,
                inline_col_feats=filter_inline_col_feats,
            )
        except Exception as e:
            print(f"Error in join: {join}", flush=True)
            raise e

    assert not (filter is not None and join is not None), (
        f"Filter and join in same plan node: {node} - the parser should prevent that! In this cases messages will get averaged. If this is wanted, please add automatic creation of a logical plan node that combines both filters and joins."
    )

    # tables
    table_id = plan_params.get("table_id")
    if (
        "Scan" in plan_params["op_name"]
        and plan_params["op_name"] != "Bitmap Index Scan"
    ):
        assert table_id is not None, f"No table in scan: {plan_params}"
    if table_id is not None:
        assert isinstance(table_id, int), (
            f"Table id is not an int: {table_id} / {plan_params}"
        )
        table_node_id = table_idx.get((table_id, run_file_id))
        run_file_table_statistics = db_statistics[run_file_id]["table_stats"]

        if table_node_id is None:
            curr_table_features = [
                encode(
                    feature_name,
                    run_file_table_statistics[table_id],
                    feature_statistics,
                )
                for feature_name in plan_featurization.TABLE_FEATURES
            ]
            table_node_id = len(table_features)
            table_features.append(curr_table_features)
            table_idx[(table_id, run_file_id)] = table_node_id

        table_to_plan_edges.append((table_node_id, plan_node_id))

    # SUMO-LOOP: Detect Nested Loop Joins and map right child to left child for loop multiplier
    nlj_children_handled = False
    if sumo_loop and len(node.get("children", [])) == 2:
        op_name_lower = plan_params.get("op_name", "").lower()
        if "nested loop" in op_name_lower:
            # Check if right child has act_loops > 1.5 (threshold for noise)
            right_child = node["children"][1]
            right_child_loops = right_child["plan_parameters"].get("act_loops", 1)

            if right_child_loops > 1.5:
                # We'll map right_child_id -> left_child_id after traversing the left subtree
                # Store current position as left child will start here
                left_child_node_id_start = len(plan_depths)
                # Traverse left child first to know its final node ID
                left_child = node["children"][0]
                plan_to_graph(
                    left_child,
                    run_file_id,
                    plan_depths,
                    plan_features,
                    subplan_labels,
                    subplan_card_labels,
                    subplan_est_card_labels,
                    plan_to_plan_edges,
                    db_statistics,
                    feature_statistics,
                    filter_to_plan_edges,
                    predicate_col_features,
                    output_column_to_plan_edges,
                    output_column_features,
                    column_to_output_column_edges,
                    column_to_filter_edges,
                    column_features,
                    table_features,
                    table_to_plan_edges,
                    output_column_idx,
                    column_idx,
                    table_idx,
                    plan_featurization,
                    predicate_depths,
                    intra_predicate_edges,
                    logical_preds,
                    parent_node_id=plan_node_id,
                    depth=depth + 1,
                    skip_top_aggregation=skip_top_aggregation,
                    filter_inline_col_feats=filter_inline_col_feats,
                    annotate_base_table_card_to_scan_op=annotate_base_table_card_to_scan_op,
                    subplan_label_op_time=subplan_label_op_time,
                    sumo_loop=sumo_loop,
                    sumo_loop_multiply_est_error=sumo_loop_multiply_est_error,
                    sumo_loop_loss_on_scaled=sumo_loop_loss_on_scaled,
                    loop_multiplier_map=loop_multiplier_map,
                )

                # Now get right child's node ID and map it
                right_child_node_id_start = len(plan_depths)
                # Traverse right child
                plan_to_graph(
                    right_child,
                    run_file_id,
                    plan_depths,
                    plan_features,
                    subplan_labels,
                    subplan_card_labels,
                    subplan_est_card_labels,
                    plan_to_plan_edges,
                    db_statistics,
                    feature_statistics,
                    filter_to_plan_edges,
                    predicate_col_features,
                    output_column_to_plan_edges,
                    output_column_features,
                    column_to_output_column_edges,
                    column_to_filter_edges,
                    column_features,
                    table_features,
                    table_to_plan_edges,
                    output_column_idx,
                    column_idx,
                    table_idx,
                    plan_featurization,
                    predicate_depths,
                    intra_predicate_edges,
                    logical_preds,
                    parent_node_id=plan_node_id,
                    depth=depth + 1,
                    skip_top_aggregation=skip_top_aggregation,
                    filter_inline_col_feats=filter_inline_col_feats,
                    annotate_base_table_card_to_scan_op=annotate_base_table_card_to_scan_op,
                    subplan_label_op_time=subplan_label_op_time,
                    sumo_loop=sumo_loop,
                    sumo_loop_multiply_est_error=sumo_loop_multiply_est_error,
                    sumo_loop_loss_on_scaled=sumo_loop_loss_on_scaled,
                    loop_multiplier_map=loop_multiplier_map,
                )

                # Map right child's top-level node to left child's top-level node
                # Right child's first node will be multiplied by left child's cardinality
                loop_multiplier_map[0].append(right_child_node_id_start)
                loop_multiplier_map[1].append(left_child_node_id_start)
                nlj_children_handled = True

    # add edge to parent
    if parent_node_id is not None:
        plan_to_plan_edges.append((plan_node_id, parent_node_id))

    # continue recursively (normal case without special NLJ handling)
    # Skip if we already handled children for NLJ
    if not nlj_children_handled:
        for i, c in enumerate(node["children"]):
            plan_to_graph(
                c,
                run_file_id,
                plan_depths,
                plan_features,
                subplan_labels,
                subplan_card_labels,
                subplan_est_card_labels,
                plan_to_plan_edges,
                db_statistics,
                feature_statistics,
                filter_to_plan_edges,
                predicate_col_features,
                output_column_to_plan_edges,
                output_column_features,
                column_to_output_column_edges,
                column_to_filter_edges,
                column_features,
                table_features,
                table_to_plan_edges,
                output_column_idx,
                column_idx,
                table_idx,
                plan_featurization,
                predicate_depths,
                intra_predicate_edges,
                logical_preds,
                parent_node_id=plan_node_id,
                depth=depth + 1,
                skip_top_aggregation=skip_top_aggregation,
                filter_inline_col_feats=filter_inline_col_feats,
                annotate_base_table_card_to_scan_op=annotate_base_table_card_to_scan_op,
                subplan_label_op_time=subplan_label_op_time,
                sumo_loop=sumo_loop,
                sumo_loop_multiply_est_error=sumo_loop_multiply_est_error,
                sumo_loop_loss_on_scaled=sumo_loop_loss_on_scaled,
                loop_multiplier_map=loop_multiplier_map,
            )
